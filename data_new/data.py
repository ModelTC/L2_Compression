import os
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader


class Imagenet():
    def __init__(self) -> None:
        super().__init__()
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def train_dataloader(self):
        transform = T.Compose(
            [   
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageFolder(root="", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=2,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [  
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageFolder(root="", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            drop_last=False,
            pin_memory=False,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()