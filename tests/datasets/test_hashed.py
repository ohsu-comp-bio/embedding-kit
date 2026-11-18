

from embkit.datasets.hashed import DatasetBloomFilter, HashedDataset, hash_sha256


class DummyHashedDatasetField(DatasetBloomFilter):
    def hash_item(self, item):
        return hash_sha256(item[0])



def test_dummy_hashed_dataset_field():
    data = [(1,), (2,), (3,)]
    dataset = DummyHashedDatasetField(data)
    for item in data:
        assert dataset.has(item)
    assert not dataset.has((4,))


def test_dummy_hashed_dataset_field_persistence(tmp_path):
    data = [(1,), (2,), (3,)]
    bloom_file = tmp_path / "dataset.bloom"
    dataset = DummyHashedDatasetField(data)
    dataset.save_bloom(bloom_file)

    restored = DummyHashedDatasetField([], bloom_file=bloom_file)

    for item in data:
        assert restored.has(item)
    assert not restored.has((4,))