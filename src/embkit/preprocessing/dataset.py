from torch.utils.data import Dataset

class DatasetMask(Dataset):
    """
    DatasetMask

    Given an input dataset, return a subset of the values of a row based on a mask
    """
    def __init__(self, dataset, mask, device=None):
        self.dataset = dataset
        self.mask = mask
        self.device = device
        self.dim = len(mask)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        o = []
        x = self.dataset[idx]
        for i in range( self.dim ):
            if self.device is not None:
                o.append( x[i][self.mask[i]].to(self.device) )
            else:
                o.append( x[i][self.mask[i]] )
        return o
