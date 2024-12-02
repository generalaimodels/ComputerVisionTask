import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from datasets import DatasetDict, load_dataset
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from statistics import mean
from transformers import SamModel, SamProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def show_mask(mask: np.ndarray, ax: plt.Axes, random_color: bool = False) -> None:
    """
    Display a mask on a matplotlib axis.

    Args:
        mask (np.ndarray): The mask to display.
        ax (plt.Axes): The matplotlib axis to display the mask on.
        random_color (bool, optional): Whether to use a random color. Defaults to False.
    """
    try:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    except Exception as e:
        logger.error(f"Error in show_mask: {e}")


def show_box(box: List[int], ax: plt.Axes, color: Tuple[float, float, float] = (0, 1, 0), lw: int = 2) -> None:
    """
    Draw a bounding box on a matplotlib axis.

    Args:
        box (List[int]): Bounding box coordinates [x_min, y_min, x_max, y_max].
        ax (plt.Axes): The matplotlib axis to draw the box on.
        color (Tuple[float, float, float], optional): Box edge color. Defaults to (0, 1, 0).
        lw (int, optional): Line width. Defaults to 2.
    """
    try:
        x0, y0, x_max, y_max = box
        width, height = x_max - x0, y_max - y0
        rect = plt.Rectangle((x0, y0), width, height, edgecolor=color, facecolor="none", linewidth=lw)
        ax.add_patch(rect)
    except Exception as e:
        logger.error(f"Error in show_box: {e}")


def show_boxes_on_image(raw_image: Image.Image, boxes: List[List[int]], figsize: Tuple[int, int] = (10, 10)) -> None:
    """
    Display an image with bounding boxes.

    Args:
        raw_image (Image.Image): The image to display.
        boxes (List[List[int]]): List of bounding boxes.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 10).
    """
    try:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        plt.imshow(raw_image)
        for box in boxes:
            show_box(box, ax)
        plt.axis('off')
        plt.show()
    except Exception as e:
        logger.error(f"Error in show_boxes_on_image: {e}")


def get_bounding_box(ground_truth_map: np.ndarray, perturbation: int = 20) -> List[int]:
    """
    Compute bounding box from a ground truth mask with optional perturbation.

    Args:
        ground_truth_map (np.ndarray): Ground truth segmentation mask.
        perturbation (int, optional): Maximum perturbation to apply to bounding box coordinates. Defaults to 20.

    Returns:
        List[int]: Bounding box coordinates [x_min, y_min, x_max, y_max].
    """
    try:
        y_indices, x_indices = np.where(ground_truth_map > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("Ground truth mask is empty.")
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, perturbation))
        x_max = min(W, x_max + np.random.randint(0, perturbation))
        y_min = max(0, y_min - np.random.randint(0, perturbation))
        y_max = min(H, y_max + np.random.randint(0, perturbation))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    except Exception as e:
        logger.error(f"Error in get_bounding_box: {e}")
        return [0, 0, 0, 0]  # Return a default invalid bounding box


class SAMDataset(Dataset):
    """
    Custom Dataset for SAM model training.
    """

    def __init__(self, dataset: DatasetDict, processor: SamProcessor) -> None:
        """
        Initialize the dataset.

        Args:
            dataset (DatasetDict): The dataset dictionary containing 'train' and 'validation' splits.
            processor (SamProcessor): The SAM model processor.
        """
        try:
            self.dataset = dataset
            self.processor = processor
        except Exception as e:
            logger.error(f"Error initializing SAMDataset: {e}")
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            item = self.dataset[idx]
            image = item["image"]
            ground_truth_mask = np.array(item["label"])

            # Normalize mask
            ground_truth_mask = ground_truth_mask / 255.0

            # Get bounding box prompt
            prompt = get_bounding_box(ground_truth_mask)

            # Prepare image and prompt for the model
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

            # Remove batch dimension
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # Add ground truth segmentation
            inputs["ground_truth_mask"] = torch.tensor(ground_truth_mask, dtype=torch.float32)

            return inputs
        except Exception as e:
            logger.error(f"Error in SAMDataset __getitem__: {e}")
            return {}


def visualize_sample(image: Image.Image, ground_truth_mask: np.ndarray, predicted_mask: Optional[np.ndarray] = None) -> None:
    """
    Visualize the image with ground truth and optionally predicted masks.

    Args:
        image (Image.Image): The original image.
        ground_truth_mask (np.ndarray): The ground truth segmentation mask.
        predicted_mask (Optional[np.ndarray], optional): The predicted mask. Defaults to None.
    """
    try:
        fig, axes = plt.subplots(1, 2 if predicted_mask is not None else 1, figsize=(15, 7))
        if predicted_mask is not None:
            axes = axes.flatten()

        # Show Ground Truth Mask
        axes[0].imshow(np.array(image))
        show_mask(ground_truth_mask, axes[0])
        axes[0].set_title("Ground Truth Mask")
        axes[0].axis("off")

        # Show Predicted Mask
        if predicted_mask is not None:
            axes[1].imshow(np.array(image))
            show_mask(predicted_mask, axes[1])
            axes[1].set_title("Predicted Mask")
            axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error in visualize_sample: {e}")


def main() -> None:
    """
    Main function to execute the training and evaluation pipeline.
    """
    try:
        # Parameters
        DATASET_NAME = "hf-vision/road-pothole-segmentation"
        MODEL_NAME = "facebook/sam-vit-base"
        NUM_EPOCHS = 10
        BATCH_SIZE = 2
        LEARNING_RATE = 1e-5
        IMAGE_SIZE = (640, 640)

        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset(DATASET_NAME)
        if not dataset:
            raise ValueError("Failed to load dataset.")

        # Initialize processor
        logger.info("Initializing processor...")
        processor = SamProcessor.from_pretrained(MODEL_NAME)

        # Initialize custom dataset
        logger.info("Preparing datasets...")
        train_dataset = SAMDataset(dataset=dataset["train"], processor=processor)
        validation_dataset = SAMDataset(dataset=dataset["validation"], processor=processor)

        # Verify sample data
        sample = train_dataset[0]
        for key, value in sample.items():
            logger.info(f"Sample Item - {key}: {value.shape if hasattr(value, 'shape') else 'N/A'}")

        # Create DataLoader
        logger.info("Creating DataLoader...")
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Load model
        logger.info("Loading model...")
        model = SamModel.from_pretrained(MODEL_NAME)

        # Freeze vision and prompt encoders
        logger.info("Freezing vision and prompt encoders...")
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad = False

        # Initialize optimizer
        logger.info("Initializing optimizer...")
        optimizer = Adam(model.mask_decoder.parameters(), lr=LEARNING_RATE, weight_decay=0)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model.to(device)

        # Training loop
        logger.info("Starting training...")
        model.train()
        for epoch in range(NUM_EPOCHS):
            epoch_losses = []
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                if not batch:
                    continue

                # Move data to device
                pixel_values = batch.get("pixel_values", torch.tensor([])).to(device)
                input_boxes = batch.get("input_boxes", torch.tensor([])).to(device)
                ground_truth_masks = batch.get("ground_truth_mask", torch.tensor([])).to(device)

                if pixel_values.numel() == 0 or input_boxes.numel() == 0 or ground_truth_masks.numel() == 0:
                    continue

                # Forward pass
                outputs = model(pixel_values=pixel_values,
                                input_boxes=input_boxes,
                                multimask_output=False)

                # Compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                predicted_masks = nn.functional.interpolate(predicted_masks,
                                                            size=IMAGE_SIZE,
                                                            mode='bilinear',
                                                            align_corners=False)
                loss = torchvision.ops.sigmoid_focal_loss(predicted_masks, ground_truth_masks.unsqueeze(1), reduction='mean')

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            mean_loss = mean(epoch_losses) if epoch_losses else 0
            logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Mean Loss: {mean_loss:.4f}")

        logger.info("Training completed.")

        # Evaluation on a validation sample
        logger.info("Evaluating on a validation sample...")
        sample_idx = 4
        val_item = dataset["validation"][sample_idx]
        image = val_item["image"]
        ground_truth_mask = np.array(val_item["label"])

        # Get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # Prepare inputs for the model
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        # Apply sigmoid to get probabilities
        sam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))

        # Resize masks to original size
        sam_seg_prob = nn.functional.interpolate(sam_seg_prob,
                                                 size=IMAGE_SIZE,
                                                 mode='bilinear',
                                                 align_corners=False)

        # Convert probabilities to binary mask
        sam_seg_prob = sam_seg_prob.cpu().numpy().squeeze()
        sam_segmentation_results = (sam_seg_prob > 0.5).astype(np.uint8)

        # Visualize results
        visualize_sample(
            image=image,
            ground_truth_mask=ground_truth_mask,
            predicted_mask=sam_segmentation_results
        )

    except Exception as e:
        logger.exception(f"An error occurred in the main pipeline: {e}")


if __name__ == "__main__":
    main()