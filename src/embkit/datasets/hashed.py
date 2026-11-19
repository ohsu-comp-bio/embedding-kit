
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from rbloom import Bloom
from hashlib import sha256


def hash_sha256(obj):
    h = sha256(str(obj).encode()).digest()
    return int.from_bytes(h[:16], "big", signed=True)

class HashedDataset(ABC, Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def hash_item(self, item):
        pass

    def __getitem__(self, idx):
        return self.dataset[idx]

class DatasetBloomFilter(HashedDataset):
    def __init__(self, dataset, bloom_file=None, bloom_capacity=1000, bloom_error_rate=0.01):
        super().__init__(dataset)
        if bloom_file is not None:
            self.bloom = Bloom.load(bloom_file, self.hash_item)
        else:
            self.bloom = Bloom(
                expected_items=bloom_capacity,
                false_positive_rate=bloom_error_rate,
                hash_func=self.hash_item,
            )
            for item in dataset:
                self.bloom.add(item)

    def has(self, item):
        return item in self.bloom

    def save_bloom(self, bloom_file):
        self.bloom.save(bloom_file)

    def __contains__(self, item):
        return self.has(item)