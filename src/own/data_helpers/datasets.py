"""Define ByteDataset class.

ByteDataset is used to load big dataset and easily shuffling. A ByteDataset is specified by 
a `idxs.pkl` file and a `data.pkl` file.

The first 4 bytes of `idxs.pkl` is size of the dataset (number of samples),
i.e. maximum number of samples = 2^32 - 1 = 4,294,967,295
Each sample is stored in a variable-length number of bytes in `data.pkl`.
The position of sample i-th (bytes offset) in `data.pkl` is specified in `idxs.pkl`,
i.e. pos = 4 + i * idx_record_size,
where `idx_record_size` is number of bytes used to specify position of a sample, meaning 
that maximum size of `data.pkl` is about 2^48 bytes = 64 TiB.
"""

import os
import pickle
from typing import Text

from torch.utils.data import Dataset


class ByteDataset(Dataset):
    def __init__(
        self,
        data_path: Text,
        idx_record_size: int,
        transform=None
    ):
        self.idx_reader = open(os.path.join(data_path, "idxs.pkl"), "rb")
        self.data_reader = open(os.path.join(data_path, "data.pkl"), "rb")
        self.idx_record_size = idx_record_size
        self.transform = transform
    
    def __len__(self):
        self.idx_reader.seek(0, 0)
        dataset_size = self.idx_reader.read(4)
        dataset_size = int.from_bytes(dataset_size, byteorder='big', signed=False)
        return dataset_size
    
    def __getitem__(self, idx):
        # get position of record
        self.idx_reader.seek(idx * self.idx_record_size + 4, 0)
        position = self.idx_reader.read(self.idx_record_size)
        position = int.from_bytes(position, 'big', signed=False)

        # get record
        self.data_reader.seek(position, 0)
        record = pickle.load(self.data_reader)

        # transform
        if self.transform:
            return self.transform(record)
        return record
