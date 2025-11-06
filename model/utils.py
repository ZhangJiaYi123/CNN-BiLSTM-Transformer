# model/utils.py
import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib
import json


def compute_metrics(y_true, y_pred, labels=None, target_names=None):
    """
    Returns dict with accuracy, precision, recall, f1 and confusion matrix.
    y_true, y_pred: 1-D numpy arrays or lists
    """
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    cls_report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f1), "report": cls_report, "confusion_matrix": cm}


def save_model(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_model(path, map_location=None):
    return torch.load(path, map_location=map_location)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_class_mapping(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_class_weights(labels, num_classes):
    """
    labels: 1D numpy array of training labels
    return torch tensor of shape (num_classes,)
    weight = total_count / (num_classes * count_i)
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    # avoid division by zero
    counts = np.where(counts == 0, 1, counts)
    weights = total / (counts * num_classes)
    return torch.tensor(weights, dtype=torch.float32)
