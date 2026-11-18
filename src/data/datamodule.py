"""Memory-bounded window dataset: store indices, not overlapping signal copies."""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.windowing import window_indices


class WindowDataset(Dataset):
    def __init__(self, records, window_seconds=5, stride_seconds=.1):
        self.records = records
        self.size = round(window_seconds * records[0].fs) if records else 0
        self.index = []
        for i, record in enumerate(records):
            if round(window_seconds * record.fs) != self.size:
                raise ValueError("All records must use the same sample rate")
            starts, labels, _ = window_indices(record, window_seconds, stride_seconds)
            self.index.extend((i, int(start), int(label)) for start, label in zip(starts, labels))
        self.targets = np.array([item[2] for item in self.index])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        recording, start, label = self.index[index]
        return (torch.from_numpy(self.records[recording].signals[start:start+self.size]),
                torch.tensor(label, dtype=torch.float32))
