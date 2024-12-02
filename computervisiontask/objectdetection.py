"""
Object Detection Pipeline using DETR on the Hardhat Dataset.

This script performs the following steps:
1. Loads the Hardhat dataset.
2. Visualizes dataset samples with bounding boxes.
3. Applies data augmentations.
4. Prepares the dataset for training.
5. Fine-tunes a DETR model for object detection.
6. Evaluates the trained model on test images.

Dependencies:
- datasets
- numpy
- Pillow
- matplotlib
- transformers
- albumentations
- torch
- requests
- tqdm

Ensure all dependencies are installed before running the script.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from datasets import Dataset, load_dataset
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
    pipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_hardhat_dataset(dataset_name: str = "anindya64/hardhat") -> Dict[str, Dataset]:
    """
    Load the Hardhat dataset.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Dict[str, Dataset]: A dictionary containing train and test datasets.
    """
    try:
        dataset = load_dataset(dataset_name)
        logger.info("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise


def draw_image_with_annotations(dataset: Dataset, idx: int) -> Image.Image:
    """
    Draw bounding boxes and labels on an image from the dataset.

    Args:
        dataset (Dataset): The dataset containing images and annotations.
        idx (int): Index of the image to draw.

    Returns:
        Image.Image: The image with drawn annotations.
    """
    try:
        sample = dataset[idx]
        image: Image.Image = sample["image"].copy()
        annotations = sample["objects"]
        draw = ImageDraw.Draw(image)
        width, height = sample["width"], sample["height"]

        for obj in zip(
            annotations["id"], annotations["category"], annotations["bbox"]
        ):
            class_id, category, bbox = obj
            x, y, w, h = bbox
            if max(bbox) > 1.0:
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
            else:
                x1 = int(x * width)
                y1 = int(y * height)
                x2 = int((x + w) * width)
                y2 = int((y + h) * height)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), category, fill="white")

        return image
    except IndexError:
        logger.error(f"Index {idx} is out of bounds for the dataset.")
        raise
    except Exception as e:
        logger.error(f"Error drawing annotations for index {idx}: {e}")
        raise


def plot_images_with_annotations(
    dataset: Dataset, indices: List[int], cols: int = 3
) -> None:
    """
    Plot multiple images with their annotations.

    Args:
        dataset (Dataset): The dataset containing images and annotations.
        indices (List[int]): List of image indices to plot.
        cols (int): Number of columns in the plot grid.
    """
    try:
        rows = (len(indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for ax, idx in zip(axes, indices):
            image = draw_image_with_annotations(dataset, idx)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"Image ID: {idx}")

        # Hide any unused subplots
        for ax in axes[len(indices) :]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting images: {e}")
        raise


def get_image_processor(checkpoint: str) -> AutoImageProcessor:
    """
    Initialize the image processor from a pretrained checkpoint.

    Args:
        checkpoint (str): The model checkpoint to use.

    Returns:
        AutoImageProcessor: The initialized image processor.
    """
    try:
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        logger.info("Image processor initialized successfully.")
        return image_processor
    except Exception as e:
        logger.error(f"Failed to initialize image processor: {e}")
        raise


def define_augmentations() -> A.Compose:
    """
    Define the data augmentations using Albumentations.

    Returns:
        A.Compose: The composed data augmentations.
    """
    try:
        transform = A.Compose(
            [
                A.Resize(480, 480),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"]),
        )
        logger.info("Data augmentations defined successfully.")
        return transform
    except Exception as e:
        logger.error(f"Failed to define augmentations: {e}")
        raise


def format_annotations(
    image_id: int, categories: List[str], areas: List[float], bboxes: List[List[float]]
) -> List[Dict[str, Any]]:
    """
    Format annotations for the model.

    Args:
        image_id (int): The image ID.
        categories (List[str]): List of category names.
        areas (List[float]): List of areas for each bounding box.
        bboxes (List[List[float]]): List of bounding boxes.

    Returns:
        List[Dict[str, Any]]: Formatted annotations.
    """
    try:
        label2id = {"head": 0, "helmet": 1, "person": 2}
        annotations = []
        for category, area, bbox in zip(categories, areas, bboxes):
            annotations.append(
                {
                    "image_id": image_id,
                    "category_id": label2id.get(category, -1),
                    "isCrowd": 0,
                    "area": area,
                    "bbox": bbox,
                }
            )
        return annotations
    except Exception as e:
        logger.error(f"Failed to format annotations for image ID {image_id}: {e}")
        raise


def apply_transformations(
    examples: Dict[str, Any], transform: A.Compose, image_processor: AutoImageProcessor
) -> Dict[str, Any]:
    """
    Apply augmentations and process images and annotations.

    Args:
        examples (Dict[str, Any]): A batch of examples from the dataset.
        transform (A.Compose): The augmentation pipeline.
        image_processor (AutoImageProcessor): The image processor.

    Returns:
        Dict[str, Any]: Transformed images and annotations.
    """
    try:
        images = []
        targets = []

        for image, objects, image_id in zip(
            examples["image"], examples["objects"], examples["image_id"]
        ):
            image_np = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB to BGR
            augmented = transform(image=image_np, bboxes=objects["bbox"], category=objects["category"])

            images.append(augmented["image"])
            annotations = format_annotations(
                image_id=image_id,
                categories=augmented["category"],
                areas=objects["area"],
                bboxes=augmented["bboxes"],
            )
            targets.append({"image_id": image_id, "annotations": annotations})

        processed = image_processor(images=images, annotations=targets, return_tensors="pt")
        return processed
    except Exception as e:
        logger.error(f"Error during transformations: {e}")
        raise


def prepare_datasets(
    dataset: Dict[str, Dataset],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
) -> Dict[str, Dataset]:
    """
    Apply transformations to train and test datasets.

    Args:
        dataset (Dict[str, Dataset]): The original train and test datasets.
        transform (A.Compose): The augmentation pipeline.
        image_processor (AutoImageProcessor): The image processor.

    Returns:
        Dict[str, Dataset]: Transformed train and test datasets.
    """
    try:
        transformed_train = dataset["train"].with_transform(
            lambda examples: apply_transformations(examples, transform, image_processor)
        )
        transformed_test = dataset["test"].with_transform(
            lambda examples: apply_transformations(examples, transform, image_processor)
        )
        logger.info("Datasets transformed successfully.")
        return {"train": transformed_train, "test": transformed_test}
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        raise


def collate_batch(batch: List[Dict[str, Any]], image_processor: AutoImageProcessor) -> Dict[str, torch.Tensor]:
    """
    Collate function to prepare batches for training.

    Args:
        batch (List[Dict[str, Any]]): A list of samples.
        image_processor (AutoImageProcessor): The image processor.

    Returns:
        Dict[str, torch.Tensor]: Batched inputs for the model.
    """
    try:
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")

        labels = [item["labels"] for item in batch]
        batch_dict = {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }
        return batch_dict
    except Exception as e:
        logger.error(f"Error during collation: {e}")
        raise


def initialize_model(
    checkpoint: str,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> AutoModelForObjectDetection:
    """
    Initialize the object detection model.

    Args:
        checkpoint (str): The model checkpoint to use.
        id2label (Dict[int, str]): Mapping from label IDs to label names.
        label2id (Dict[str, int]): Mapping from label names to label IDs.

    Returns:
        AutoModelForObjectDetection: The initialized model.
    """
    try:
        model = AutoModelForObjectDetection.from_pretrained(
            checkpoint,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        logger.info("Model initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def get_training_arguments(output_dir: str) -> TrainingArguments:
    """
    Define the training arguments.

    Args:
        output_dir (str): Directory to save model checkpoints and logs.

    Returns:
        TrainingArguments: The training configuration.
    """
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            num_train_epochs=3,
            max_steps=1000,
            fp16=True,
            save_steps=100,
            logging_steps=30,
            learning_rate=1e-5,
            weight_decay=1e-4,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
        )
        logger.info("Training arguments defined successfully.")
        return training_args
    except Exception as e:
        logger.error(f"Failed to get training arguments: {e}")
        raise


def create_trainer(
    model: AutoModelForObjectDetection,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    image_processor: AutoImageProcessor,
) -> Trainer:
    """
    Initialize the Trainer.

    Args:
        model (AutoModelForObjectDetection): The object detection model.
        training_args (TrainingArguments): Training configuration.
        train_dataset (Dataset): The transformed training dataset.
        eval_dataset (Dataset): The transformed evaluation dataset.
        image_processor (AutoImageProcessor): The image processor.

    Returns:
        Trainer: The initialized trainer.
    """
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=lambda batch: collate_batch(batch, image_processor),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=image_processor,
        )
        logger.info("Trainer created successfully.")
        return trainer
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        raise


def train_model(trainer: Trainer) -> None:
    """
    Train the model.

    Args:
        trainer (Trainer): The Trainer instance.
    """
    try:
        trainer.train()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def download_image(url: str) -> Image.Image:
    """
    Download an image from a URL.

    Args:
        url (str): The URL of the image.

    Returns:
        Image.Image: The downloaded image.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        logger.info(f"Image downloaded successfully from {url}.")
        return image
    except requests.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing the downloaded image: {e}")
        raise


def initialize_object_detection_pipeline(
    model_name: str,
) -> pipeline:
    """
    Initialize the object detection pipeline.

    Args:
        model_name (str): The model name to use for the pipeline.

    Returns:
        pipeline: The object detection pipeline.
    """
    try:
        obj_detector = pipeline("object-detection", model=model_name)
        logger.info("Object detection pipeline initialized successfully.")
        return obj_detector
    except Exception as e:
        logger.error(f"Failed to initialize object detection pipeline: {e}")
        raise


def plot_prediction_results(
    image: Image.Image, results: List[Dict[str, Any]], threshold: float = 0.7
) -> Image.Image:
    """
    Plot the prediction results on the image.

    Args:
        image (Image.Image): The original image.
        results (List[Dict[str, Any]]): The detection results.
        threshold (float): Confidence threshold for displaying boxes.

    Returns:
        Image.Image: The image with plotted predictions.
    """
    try:
        draw = ImageDraw.Draw(image)
        for result in results:
            score = result.get("score", 0)
            label = result.get("label", "N/A")
            box = result.get("box", {})
            if score > threshold:
                x, y, x2, y2 = box.values()
                draw.rectangle([x, y, x2, y2], outline="red", width=2)
                draw.text((x, y - 10), f"{label}: {score:.2f}", fill="yellow")
        return image
    except Exception as e:
        logger.error(f"Error plotting prediction results: {e}")
        raise


def predict_and_plot(
    image: Image.Image, obj_detector: pipeline, threshold: float = 0.7
) -> Image.Image:
    """
    Perform prediction on an image and plot the results.

    Args:
        image (Image.Image): The input image.
        obj_detector (pipeline): The object detection pipeline.
        threshold (float): Confidence threshold for displaying boxes.

    Returns:
        Image.Image: The image with prediction results plotted.
    """
    try:
        results = obj_detector(image)
        plotted_image = plot_prediction_results(image, results, threshold)
        return plotted_image
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def plot_test_predictions(
    dataset: Dataset, indices: List[int], obj_detector: pipeline, cols: int = 3
) -> None:
    """
    Plot test images with prediction results.

    Args:
        dataset (Dataset): The test dataset.
        indices (List[int]): List of image indices to plot.
        obj_detector (pipeline): The object detection pipeline.
        cols (int): Number of columns in the plot grid.
    """
    try:
        rows = (len(indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for ax, idx in zip(axes, indices):
            image = dataset[idx]["image"].copy()
            predicted_image = predict_and_plot(image, obj_detector)
            ax.imshow(predicted_image)
            ax.axis("off")
            ax.set_title(f"Prediction ID: {idx}")

        # Hide any unused subplots
        for ax in axes[len(indices) :]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting test predictions: {e}")
        raise


def main() -> None:
    """
    Main function to execute the object detection pipeline.
    """
    try:
        # Load dataset
        dataset = load_hardhat_dataset()

        # Visualize some training samples
        plot_images_with_annotations(dataset["train"], indices=list(range(6)))

        # Initialize image processor
        checkpoint = "facebook/detr-resnet-50-dc5"
        image_processor = get_image_processor(checkpoint)

        # Define data augmentations
        augmentations = define_augmentations()

        # Prepare transformed datasets
        transformed_datasets = prepare_datasets(dataset, augmentations, image_processor)
        train_transformed = transformed_datasets["train"]
        test_transformed = transformed_datasets["test"]

        # Define label mappings
        id2label = {0: "head", 1: "helmet", 2: "person"}
        label2id = {v: k for k, v in id2label.items()}

        # Initialize model
        model = initialize_model(checkpoint, id2label, label2id)

        # Define training arguments
        training_args = get_training_arguments(output_dir="detr-resnet-50-hardhat-finetuned")

        # Create trainer
        trainer = create_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_transformed,
            eval_dataset=test_transformed,
            image_processor=image_processor,
        )

        # Train the model
        train_model(trainer)

        # Initialize object detection pipeline
        trained_model_name = "detr-resnet-50-hardhat-finetuned"  # Update with your model's name
        obj_detector = initialize_object_detection_pipeline(trained_model_name)

        # Predict on a sample image
        sample_image_url = (
            "https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/test-helmet-object-detection.jpg"
        )
        sample_image = download_image(sample_image_url)
        predicted_image = predict_and_plot(sample_image, obj_detector)
        predicted_image.show()

        # Plot predictions on test dataset
        plot_test_predictions(
            dataset["test"], indices=list(range(6)), obj_detector=obj_detector
        )

    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    main()