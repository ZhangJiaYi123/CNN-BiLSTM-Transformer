# model/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CICIDSWindowDataset(Dataset):
    """
    Loads preprocessed windows from a single .npz created by preprocess/make_windows.py
    Expects keys: train_X/train_y / val_X/val_y / test_X/test_y OR generic X & y passed externally.
    Each X window: shape (T, F) as float32; labels are ints in [0, C-1].
    """

    def __init__(self, npz_path, split='train', transforms=None):
        assert split in (
            'train', 'val', 'test'), "split must be 'train'|'val'|'test'"
        self.npz_path = npz_path
        self.split = split
        self.transforms = transforms

        data = np.load(npz_path, allow_pickle=True)
        # support both combined file and separated naming
        key_X = f"{split}_X"
        key_y = f"{split}_y"
        if key_X not in data:
            # older naming: train_X etc. if keys differ, try fallback:
            available = list(data.keys())
            raise KeyError(
                f"{key_X} not found in {npz_path}. Available keys: {available}")
        X = data[key_X]    # (N, T, F)
        y = data[key_y]    # (N,)
        # ensure types
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        # store feature info if present
        self.feature_names = data.get('feature_names', None)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = self.X[idx]    # (T, F)
        y = int(self.y[idx])
        if self.transforms:
            x = self.transforms(x)
        # convert to tensors: shape (T, F)
        x = torch.from_numpy(x)  # float32
        return x, torch.tensor(y, dtype=torch.long)


def make_dataloader(npz_path, split='train', batch_size=64, shuffle=True, num_workers=4, pin_memory=True):
    ds = CICIDSWindowDataset(npz_path, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader
