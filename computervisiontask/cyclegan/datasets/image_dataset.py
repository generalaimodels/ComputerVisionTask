# datasets/image_dataset.py

import os
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import datasets  # Ensure you have the `datasets` library installed


class ImageDataset(Dataset):
    """
    Custom Dataset for loading and processing images for CycleGAN.

    Attributes:
        files_A (List[Image]): Images from domain A.
        files_B (List[Image]): Images from domain B.
        transform (Callable): Transformations to apply to the images.
        randperm (torch.Tensor): Random permutation for pairing images.
    """

    def __init__(
        self,
        dataset: datasets.Dataset,
        transform: Optional[Callable] = None,
        mode: str = 'train'
    ) -> None:
        self.transform = transform
        self.files_A = dataset.filter(lambda x: x['label'] == 2)["image"]
        self.files_B = dataset.filter(lambda x: x['label'] == 3)["image"]
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise ValueError("Dataset must contain images for both domains A and B.")

    def new_perm(self) -> None:
        """
        Generates a new random permutation for pairing images from domain B.
        """
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves a pair of images from domains A and B.

        Args:
            index (int): Index of the image pair.

        Returns:
            tuple: Transformed images from domain A and B.
        """
        try:
            item_A = self.transform(self.files_A[index % len(self.files_A)])
            item_B = self.transform(self.files_B[self.randperm[index]])
            
            if item_A.shape[0] != 3:
                item_A = item_A.repeat(3, 1, 1)
            if item_B.shape[0] != 3:
                item_B = item_B.repeat(3, 1, 1)
            
            if index == len(self) - 1:
                self.new_perm()
            
            return (item_A - 0.5) * 2, (item_B - 0.5) * 2
        except Exception as e:
            raise RuntimeError(f"Error processing index {index}: {e}")

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Number of image pairs.
        """
        return min(len(self.files_A), len(self.files_B))