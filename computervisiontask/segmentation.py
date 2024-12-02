import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import evaluate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)

from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    seed: int = 42
    hf_dataset_id: str = "segments/sidewalk-semantic"
    split_test_size: float = 0.2
    split_val_size: float = 0.05
    id2label_filename: str = "id2label.json"
    batch_size: int = 2
    num_workers: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 100
    log_interval: int = 100
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    max_batches_evaluate: Optional[int] = None
    mean: List[float] = field(
        default_factory=lambda: [123.675, 116.280, 103.530]
    )
    std: List[float] = field(
        default_factory=lambda: [58.395, 57.120, 57.375]
    )


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def load_id2label(config: Config) -> Dict[int, str]:
    """
    Load the id2label mapping from the Hugging Face Hub.

    Args:
        config (Config): Configuration object.

    Returns:
        Dict[int, str]: Mapping from label IDs to label names.
    """
    try:
        filepath = hf_hub_download(
            config.hf_dataset_id,
            config.id2label_filename,
            repo_type="dataset",
        )
        with open(filepath, "r") as f:
            id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
        logger.info("id2label mapping loaded successfully.")
        return id2label
    except Exception as e:
        logger.error(f"Failed to load id2label mapping: {e}")
        raise


def prepare_datasets(config: Config) -> Tuple:
    """
    Load and split the dataset into training, validation, and testing sets.

    Args:
        config (Config): Configuration object.

    Returns:
        Tuple: train, validation, and test datasets.
    """
    try:
        dataset = load_dataset(config.hf_dataset_id)
        dataset = dataset.shuffle(seed=config.seed)
        dataset = dataset["train"].train_test_split(test_size=config.split_test_size)
        train_ds, test_ds = dataset["train"], dataset["test"]

        train_val_split = train_ds.train_test_split(test_size=config.split_val_size, seed=config.seed)
        train_ds, val_ds = train_val_split["train"], train_val_split["test"]

        logger.info("Datasets loaded and split successfully.")
        return train_ds, val_ds, test_ds
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        raise


def create_label_mappings(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label-to-ID and ID-to-label mappings.

    Args:
        labels (List[str]): List of label names.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: label2id and id2label mappings.
    """
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def initialize_transforms(config: Config) -> Tuple[A.Compose, A.Compose]:
    """
    Initialize training and testing transformations.

    Args:
        config (Config): Configuration object.

    Returns:
        Tuple[A.Compose, A.Compose]: Training and testing transformations.
    """
    ade_mean = np.array(config.mean) / 255.0
    ade_std = np.array(config.std) / 255.0

    train_transform = A.Compose([
        A.RandomCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ade_mean, std=ade_std),
    ])

    test_transform = A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize(mean=ade_mean, std=ade_std),
    ])

    logger.info("Transforms initialized.")
    return train_transform, test_transform


@dataclass
class SegmentationDataInput:
    original_image: np.ndarray
    transformed_image: np.ndarray
    original_segmentation_map: np.ndarray
    transformed_segmentation_map: np.ndarray


class SemanticSegmentationDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: A.Compose) -> None:
        """
        Dataset for Semantic Segmentation.

        Args:
            dataset (Dataset): A dataset containing images and segmentation maps.
            transform (A.Compose): Transformation to apply.
        """
        if 'pixel_values' not in dataset.column_names or 'label' not in dataset.column_names:
            logger.error("Dataset must contain 'pixel_values' and 'label' columns.")
            raise ValueError("Dataset must contain 'pixel_values' and 'label' columns.")

        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> SegmentationDataInput:
        try:
            sample = self.dataset[idx]
            original_image = np.array(sample["pixel_values"])
            original_segmentation_map = np.array(sample["label"])

            transformed = self.transform(image=original_image, mask=original_segmentation_map)
            transformed_image = transformed["image"].transpose(2, 0, 1)  # To channel-first
            transformed_segmentation_map = transformed["mask"]

            return SegmentationDataInput(
                original_image=original_image,
                transformed_image=transformed_image,
                original_segmentation_map=original_segmentation_map,
                transformed_segmentation_map=transformed_segmentation_map,
            )
        except Exception as e:
            logger.error(f"Error fetching item at index {idx}: {e}")
            raise


def collate_fn(batch: List[SegmentationDataInput], preprocessor: AutoImageProcessor) -> Dict[str, Any]:
    """
    Custom collate function to batch data samples.

    Args:
        batch (List[SegmentationDataInput]): List of data samples.
        preprocessor (AutoImageProcessor): Preprocessor for MaskFormer.

    Returns:
        Dict[str, Any]: Batched data.
    """
    try:
        transformed_images = [sample.transformed_image for sample in batch]
        transformed_segmentation_maps = [sample.transformed_segmentation_map for sample in batch]
        original_images = [sample.original_image for sample in batch]
        original_segmentation_maps = [sample.original_segmentation_map for sample in batch]

        preprocessed_batch = preprocessor(
            transformed_images,
            segmentation_maps=transformed_segmentation_maps,
            return_tensors="pt",
        )

        preprocessed_batch["original_images"] = original_images
        preprocessed_batch["original_segmentation_maps"] = original_segmentation_maps

        return preprocessed_batch
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}")
        raise


def denormalize_image(image: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    """
    Denormalizes a normalized image.

    Args:
        image (torch.Tensor): Normalized image tensor.
        mean (List[float]): Mean used for normalization.
        std (List[float]): Standard deviation used for normalization.

    Returns:
        np.ndarray: Denormalized image as a NumPy array.
    """
    try:
        unnormalized = image.clone().detach().cpu().numpy()
        for c in range(3):
            unnormalized[c] = (unnormalized[c] * std[c] + mean[c]) * 255.0
        unnormalized = unnormalized.astype(np.uint8).transpose(1, 2, 0)
        return unnormalized
    except Exception as e:
        logger.error(f"Error in denormalizing image: {e}")
        raise


def show_samples(dataset: Dataset, n: int = 5) -> None:
    """
    Displays 'n' samples from the dataset.

    Args:
        dataset (Dataset): The dataset containing 'pixel_values' and 'label'.
        n (int): Number of samples to display.
    """
    if n > len(dataset):
        logger.error("Requested number of samples exceeds dataset size.")
        raise ValueError("n is larger than the dataset size")

    try:
        fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n))
        for i in range(n):
            sample = dataset[i]
            image = np.array(sample["pixel_values"])
            label = np.array(sample["label"])

            axs[i, 0].imshow(image)
            axs[i, 0].set_title("Image")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(image)
            axs[i, 1].imshow(label, cmap="nipy_spectral", alpha=0.5)
            axs[i, 1].set_title("Segmentation Map")
            axs[i, 1].axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error in show_samples: {e}")
        raise


def visualize_prediction(
    image: Image.Image,
    segmentation_map: np.ndarray,
    id2label: Dict[int, str],
    alpha: float = 0.5
) -> None:
    """
    Visualizes the predicted segmentation map overlayed on the image.

    Args:
        image (Image.Image): The original image.
        segmentation_map (np.ndarray): The predicted segmentation map.
        id2label (Dict[int, str]): Mapping from label IDs to label names.
        alpha (float): Transparency factor for the overlay.
    """
    try:
        num_classes = len(np.unique(segmentation_map))
        cmap = plt.cm.get_cmap("hsv", num_classes)

        overlay = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 4))
        for i, unique_value in enumerate(np.unique(segmentation_map)):
            overlay[segmentation_map == unique_value, :3] = cmap(i)[:3]
            overlay[segmentation_map == unique_value, 3] = alpha

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)
        ax.imshow(overlay, interpolation="nearest")
        plt.axis("off")
        plt.show()
    except Exception as e:
        logger.error(f"Error in visualize_prediction: {e}")
        raise


class SemanticSegmentationModel:
    def __init__(self, config: Config, id2label: Dict[int, str]) -> None:
        """
        Initializes the MaskFormer model for semantic segmentation.

        Args:
            config (Config): Configuration object.
            id2label (Dict[int, str]): Mapping from label IDs to label names.
        """
        self.config = config
        self.id2label = id2label
        try:
            self.processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
            self.model = MaskFormerForInstanceSegmentation.from_pretrained(
                "facebook/maskformer-swin-base-ade",
                id2label=id2label,
                ignore_mismatched_sizes=True
            )
            # Freeze pixel level module
            for param in self.model.model.pixel_level_module.parameters():
                param.requires_grad = False
            logger.info("Model initialized and pixel level module frozen.")
        except Exception as e:
            logger.error(f"Failed to initialize the model: {e}")
            raise

    def to_device(self) -> None:
        """Moves the model to the specified device."""
        self.model.to(self.config.device)
        logger.info(f"Model moved to {self.config.device}.")

    def evaluate(
        self,
        dataloader: DataLoader,
        metric: Any,
        max_batches: Optional[int] = None
    ) -> float:
        """
        Evaluates the model on the given dataloader.

        Args:
            dataloader (DataLoader): DataLoader for evaluation.
            metric (Any): Evaluation metric.
            max_batches (Optional[int]): Maximum number of batches to evaluate.

        Returns:
            float: Mean IoU score.
        """
        self.model.eval()
        metric.reset()
        running_iou = 0.0
        num_batches = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if max_batches and idx >= max_batches:
                    break

                pixel_values = batch["pixel_values"].to(self.config.device)
                outputs = self.model(pixel_values=pixel_values)

                target_sizes = [
                    (image.shape[0], image.shape[1]) for image in batch["original_images"]
                ]

                predicted_semantic_maps = self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )

                ground_truth_maps = batch["original_segmentation_maps"]

                metric.add_batch(
                    predictions=predicted_semantic_maps,
                    references=ground_truth_maps
                )

                current_iou = metric.compute(num_labels=len(self.id2label), ignore_index=0)["mean_iou"]
                running_iou += current_iou
                num_batches += 1

        mean_iou = running_iou / num_batches if num_batches > 0 else 0.0
        logger.info(f"Evaluation completed. Mean IoU: {mean_iou:.4f}")
        return mean_iou

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metric: Any
    ) -> None:
        """
        Trains the model.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
            metric (Any): Evaluation metric.
        """
        self.to_device()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            self.model.train()
            running_loss = 0.0
            num_samples = 0

            for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                optimizer.zero_grad()

                try:
                    outputs = self.model(
                        pixel_values=batch["pixel_values"].to(self.config.device),
                        mask_labels=batch.get("mask_labels"),
                        class_labels=batch.get("class_labels"),
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_samples += batch["pixel_values"].size(0)

                    if idx % self.config.log_interval == 0 and idx > 0:
                        avg_loss = running_loss / num_samples
                        logger.info(f"Batch {idx} - Loss: {avg_loss:.4f}")

                except Exception as e:
                    logger.error(f"Error during training at batch {idx}: {e}")
                    continue  # Skip to the next batch

            # Evaluate on validation set
            val_mean_iou = self.evaluate(
                val_dataloader,
                metric,
                max_batches=self.config.max_batches_evaluate
            )
            logger.info(f"Epoch {epoch + 1} completed. Validation Mean IoU: {val_mean_iou:.4f}")


def main():
    # Initialize configuration
    config = Config()

    # Set seed for reproducibility
    set_seed(config.seed)

    # Load label mappings
    id2label = load_id2label(config)
    label2id = {label: idx for idx, label in id2label.items()}

    # Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(config)

    # Initialize transformations
    train_transform, test_transform = initialize_transforms(config)

    # Create datasets
    train_dataset = SemanticSegmentationDataset(train_ds, transform=train_transform)
    val_dataset = SemanticSegmentationDataset(val_ds, transform=test_transform)
    test_dataset = SemanticSegmentationDataset(test_ds, transform=test_transform)

    # Initialize preprocessor
    preprocessor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")

    # Prepare data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, preprocessor)
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, preprocessor)
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, preprocessor)
    )

    # Initialize evaluation metric
    metric = evaluate.load("mean_iou")

    # Initialize the model
    model = SemanticSegmentationModel(config, id2label)

    # Train the model
    model.train(train_dataloader, val_dataloader, metric)

    # Optionally, evaluate on the test set
    # test_mean_iou = model.evaluate(test_dataloader, metric)
    # logger.info(f"Test Mean IoU: {test_mean_iou:.4f}")

    # Visualize some predictions
    sample = test_dataset[0]
    denormalized = denormalize_image(
        torch.tensor(sample.transformed_image),
        config.mean, config.std
    )
    pil_image = Image.fromarray(denormalized)
    visualize_prediction(pil_image, sample.transformed_segmentation_map, id2label)


if __name__ == "__main__":
    main()