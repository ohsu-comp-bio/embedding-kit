from torch.utils.data import Dataset

class DatasetMask(Dataset):
    """
    DatasetMask

    Given an input dataset, return a subset of the values of a row based on a mask
    """
    def __init__(self, dataset, mask):
        self.dataset = dataset
        self.mask = mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][self.mask]

class DatasetArray(Dataset):
    """
    DatasetArray

    Given an input dataset, return all the values wrapped as an array
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return [ self.dataset[idx] ]