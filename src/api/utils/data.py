from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

import torch
from tqdm.auto import tqdm
import os


# create custom dataset for Wikipedia dataset
class WikiDataset(Dataset):

    def __init__(
        self,
        dataset_name: str,
        dataset_version: str,
        num_samples: int,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.num_samples = num_samples
        # download dataset
        self.dataset = load_dataset(
            self.dataset_name, self.dataset_version, split="train"
        )
        # shuffle to fetch random samples
        self.dataset = self.dataset.shuffle(seed=42)

        # keep only a number of samples less storage usage
        self.data = self.dataset[:num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        text = self.data["text"][index]
        id = self.data["id"][index]
        source = self.data["url"][index]
        title = self.data["title"][index]
        meta = {"sample_id": str(id), "text": text, "title": title, "source": source}
        return meta


def create_dataloader(
    dataset: WikiDataset, batch_size: int, num_workers: int = 0
) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataloader
