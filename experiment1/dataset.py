import torch as T
from torch.utils.data import Dataset


class NumberAdd(Dataset):
    """Dataset for adding numbers"""
    def __init__(self, total_samples: int) -> None:
        """Generate dataset with pairs of numbers and their sum"""
        self.N = total_samples
        self.samples = T.randint(low=-1000, high=1000, size=(self.N, 2))
        self.targets = T.sum(self.samples, dim=-1)

    def __getitem__(self, index: int) -> dict:
        """Get a pair of sample / target"""
        sample = self.samples[index]
        target = self.targets[index]
        return {"sample": sample, "target": target}

    def __len__(self) -> int:
        """Size of the dataset"""
        return self.N


if __name__ == "__main__":
    ds = NumberAdd(total_samples=100)
    for i in range(len(ds)):
        data = ds[i]
        print(data["sample"], data["target"])
        assert T.sum(data["sample"]) == data["target"]