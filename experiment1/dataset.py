import torch as T
from torch.utils.data import Dataset


class NumberAdd(Dataset):
    """Dataset for adding numbers"""
    def __init__(self, total_samples: int) -> None:
        """Generate dataset with pairs of numbers and their sum"""
        self.N = total_samples
        self.samples = T.randint(low=-1000, high=1000, size=(self.N, 2), dtype=T.float32)
        self.targets = T.sum(self.samples, dim=-1)

    def __getitem__(self, index: int) -> tuple:
        """Get a pair of sample / target"""
        sample = self.samples[index]
        target = self.targets[index]
        return sample, target

    def __len__(self) -> int:
        """Size of the dataset"""
        return self.N


if __name__ == "__main__":
    ds = NumberAdd(total_samples=100)
    for i in range(len(ds)):
        sample, target = ds[i]
        print(sample, target)
        assert T.sum(sample) == target