# model/train.py
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import CICIDSWindowDataset
from model import CNN_BiLSTM_Transformer
from utils import compute_metrics, save_model, compute_class_weights, load_class_mapping


def evaluate(model, loader, device):
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            ys.append(yb.cpu().numpy())
            preds.append(pred.cpu().numpy())
    ys = np.concatenate(ys, axis=0)
    preds = np.concatenate(preds, axis=0)
    return ys, preds


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          and args.device == "cuda" else "cpu")
    print("Using device:", device)

    # Load data
    npz_path = args.npz_path
    # Create dataset/dataloader
    train_ds = CICIDSWindowDataset(npz_path, split='train')
    val_ds = CICIDSWindowDataset(npz_path, split='val')
    test_ds = CICIDSWindowDataset(npz_path, split='test')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model instantiation
    n_features = train_ds.X.shape[2]
    num_classes = int(np.max(train_ds.y) + 1)
    print(f"n_features={n_features}, num_classes={num_classes}")
    model = CNN_BiLSTM_Transformer(
        n_features=n_features,
        cnn_channels=args.cnn_channels,
        cnn_kernel_size=args.cnn_kernel_size,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        transformer_dmodel=args.transformer_dmodel,
        transformer_nhead=args.transformer_nhead,
        transformer_layers=args.transformer_layers,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    # Loss: weighted cross entropy using class weights computed from training labels
    class_weights = compute_class_weights(train_ds.y, num_classes)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_val_f1 = 0.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)   # (batch, T, F)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        # Validation
        y_val, y_val_pred = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_val, y_val_pred)
        val_f1 = metrics['f1']
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_f1={val_f1:.4f} | val_acc={metrics['accuracy']:.4f} | time={elapsed:.1f}s")

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(
                args.checkpoint_dir, f"best_model_epoch{epoch}_f1{best_val_f1:.4f}.pt")
            save_model({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args)
            }, best_path)
            print("Saved best model ->", best_path)

    # Load best model for final test (if saved)
    if best_path is not None:
        print("Loading best model from", best_path)
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    # Final test
    y_test, y_test_pred = evaluate(model, test_loader, device)
    final_metrics = compute_metrics(y_test, y_test_pred)
    print("=== Final Test Results ===")
    print("Accuracy:", final_metrics['accuracy'])
    print("Precision/Recall/F1 (weighted):",
          final_metrics['precision'], final_metrics['recall'], final_metrics['f1'])
    print("Classification report:\n", final_metrics['report'])
    print("Confusion matrix:\n", final_metrics['confusion_matrix'])

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    save_model({
        "model_state_dict": model.state_dict(),
        "args": vars(args)
    }, final_path)
    print("Saved final model ->", final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str,
                        default="../outputs/cicids2017_windows.npz", help="Path to preprocessed .npz")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="../outputs/checkpoints", help="Where to save models")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--device", type=str,
                        default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cnn_channels", type=int, default=64)
    parser.add_argument("--cnn_kernel_size", type=int, default=3)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--transformer_dmodel", type=int, default=128)
    parser.add_argument("--transformer_nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # adjust npz path if called from project root
    # If user runs from project root, the default relative paths might need adjustments
    train_loop(args)
