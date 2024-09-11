from abc import ABC

import torch
from torch_geometric.data import Dataset


class DatasetSDP(Dataset, ABC):

    def __init__(self, data):
        super(DatasetSDP, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def get(self, idx):
        return self.data_list[idx]
