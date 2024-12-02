import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from datasets import load_dataset
from PIL import Image
import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SamProcessor, SamModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SAMDataset(Dataset):
    """Custom Dataset for SAM model."""

    def __init__(self, dataset: Any, processor: SamProcessor) -> None:
        """
        Initialize the dataset.

        Args:
            dataset (Any): The dataset loaded from Hugging Face.
            processor (SamProcessor): Processor for the SAM model.
        """
        self.dataset = dataset
        self.processor = processor

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            Dict[str, Any]: Processed inputs for the model.
        """
        try:
            item = self.dataset[idx]
            image = item["image"]
            ground_truth_mask = np.array(item["label"])

            # Normalize mask
            ground_truth_mask = ground_truth_mask / 255.0

            # Get bounding box prompt
            prompt = get_bounding_box(ground_truth_mask)

            # Prepare image and prompt for the model
            inputs = self.processor(
                images=image,
                input_boxes=[[prompt]],
                return_tensors="pt"
            )

            # Remove batch dimension
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # Add ground truth segmentation
            inputs["ground_truth_mask"] = torch.tensor(ground_truth_mask, dtype=torch.float32)

            return inputs
        except Exception as e:
            logger.error(f"Error in __getitem__ at index {idx}: {e}")
            raise


def get_bounding_box(ground_truth_map: np.ndarray) -> List[int]:
    """
    Get bounding box from mask with perturbation.

    Args:
        ground_truth_map (np.ndarray): Ground truth segmentation mask.

    Returns:
        List[int]: Perturbed bounding box coordinates [x_min, y_min, x_max, y_max].
    """
    try:
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        perturb = lambda coord_min, coord_max, max_val: [
            max(0, coord_min - np.random.randint(0, 20)),
            min(max_val, coord_max + np.random.randint(0, 20))
        ]

        x_min, x_max = perturb(x_min, x_max, W)
        y_min, y_max = perturb(y_min, y_max, H)

        return [x_min, y_min, x_max, y_max]
    except Exception as e:
        logger.error(f"Error in get_bounding_box: {e}")
        raise


def show_mask_plotly(mask: np.ndarray, image: Image.Image, title: str) -> go.Figure:
    """
    Create a Plotly figure with the mask overlayed on the image.

    Args:
        mask (np.ndarray): Binary mask to overlay.
        image (Image.Image): Original image.
        title (str): Title of the plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    try:
        fig = go.Figure()

        # Add the image
        fig.add_trace(
            go.Image(z=image)
        )

        # Create mask RGB
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        mask_rgb[..., :3] = [30 / 255, 144 / 255, 255 / 255]  # DodgerBlue
        mask_rgb[..., 3] = 0.6  # Alpha channel

        # Apply mask
        mask_rgb[mask > 0.5] = [30 / 255, 144 / 255, 255 / 255, 0.6]

        fig.add_trace(
            go.Image(z=mask_rgb)
        )

        fig.update_layout(title=title, width=640, height=640)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        return fig
    except Exception as e:
        logger.error(f"Error in show_mask_plotly: {e}")
        raise


def save_plotly_figure(fig: go.Figure, filepath: str) -> None:
    """
    Save a Plotly figure as an HTML file.

    Args:
        fig (go.Figure): Plotly figure to save.
        filepath (str): Path to save the HTML file.
    """
    try:
        fig.write_html(filepath)
        logger.info(f"Plot saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving plot to {filepath}: {e}")
        raise


def visualize_sample(
    image: Image.Image,
    ground_truth_mask: np.ndarray,
    predicted_mask: Optional[np.ndarray] = None,
    save_dir: str = "plots",
    sample_idx: int = 0
) -> None:
    """
    Visualize and save ground truth and predicted masks.

    Args:
        image (Image.Image): Original image.
        ground_truth_mask (np.ndarray): Ground truth mask.
        predicted_mask (Optional[np.ndarray], optional): Predicted mask. Defaults to None.
        save_dir (str, optional): Directory to save plots. Defaults to "plots".
        sample_idx (int, optional): Sample index for naming files. Defaults to 0.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Ground truth mask
        gt_fig = show_mask_plotly(ground_truth_mask, image, "Ground Truth Mask")
        save_plotly_figure(gt_fig, os.path.join(save_dir, f"ground_truth_mask_{sample_idx}.html"))

        if predicted_mask is not None:
            pred_fig = show_mask_plotly(predicted_mask, image, "Predicted Mask")
            save_plotly_figure(pred_fig, os.path.join(save_dir, f"predicted_mask_{sample_idx}.html"))
    except Exception as e:
        logger.error(f"Error in visualize_sample: {e}")
        raise


def train_model(
    model: SamModel,
    dataloader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    num_epochs: int = 10
) -> List[float]:
    """
    Train the model.

    Args:
        model (SamModel): The SAM model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Adam): Optimizer.
        device (torch.device): Device to train on.
        num_epochs (int, optional): Number of epochs. Defaults to 10.

    Returns:
        List[float]: List of mean losses per epoch.
    """
    try:
        model.train()
        epoch_losses = []

        for epoch in range(num_epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
            batch_losses = []

            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                try:
                    pixel_values = batch["pixel_values"].to(device)
                    input_boxes = batch["input_boxes"].to(device)
                    ground_truth_masks = batch["ground_truth_mask"].to(device)

                    optimizer.zero_grad()

                    outputs = model(
                        pixel_values=pixel_values,
                        input_boxes=input_boxes,
                        multimask_output=False
                    )

                    # Resize predicted masks
                    predicted_masks = nn.functional.interpolate(
                        outputs.pred_masks,
                        size=ground_truth_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)

                    # Compute loss
                    loss_fn = torchvision.ops.sigmoid_focal_loss
                    loss = loss_fn(
                        predicted_masks,
                        ground_truth_masks.unsqueeze(1),
                        reduction='mean'
                    )

                    loss.backward()
                    optimizer.step()

                    batch_losses.append(loss.item())
                except Exception as batch_e:
                    logger.error(f"Error during training step: {batch_e}")
                    continue

            mean_loss = np.mean(batch_losses)
            epoch_losses.append(mean_loss)
            logger.info(f"Epoch {epoch + 1} - Mean Loss: {mean_loss:.4f}")

        return epoch_losses
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise


def evaluate_model(
    model: SamModel,
    dataset: Dataset,
    processor: SamProcessor,
    device: torch.device,
    save_dir: str = "plots",
    num_samples: int = 5
) -> None:
    """
    Evaluate the model on the validation set and save visualizations.

    Args:
        model (SamModel): The trained SAM model.
        dataset (Dataset): Validation dataset.
        processor (SamProcessor): Processor for the SAM model.
        device (torch.device): Device to perform inference.
        save_dir (str, optional): Directory to save plots. Defaults to "plots".
        num_samples (int, optional): Number of samples to evaluate. Defaults to 5.
    """
    try:
        model.eval()
        os.makedirs(save_dir, exist_ok=True)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        samples_processed = 0

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if samples_processed >= num_samples:
                    break

                try:
                    image = dataset[idx]["image"]
                    ground_truth_mask = dataset[idx]["label"]

                    inputs = {
                        "pixel_values": batch["pixel_values"].to(device),
                        "input_boxes": batch["input_boxes"].to(device)
                    }

                    outputs = model(**inputs, multimask_output=False)
                    sam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
                    sam_seg_prob = nn.functional.interpolate(
                        sam_seg_prob,
                        size=(640, 640),
                        mode='bilinear',
                        align_corners=False
                    )
                    sam_seg_prob = sam_seg_prob.cpu().numpy().squeeze()
                    predicted_mask = (sam_seg_prob > 0.5).astype(np.uint8)

                    visualize_sample(
                        image=image,
                        ground_truth_mask=np.array(ground_truth_mask),
                        predicted_mask=predicted_mask,
                        save_dir=save_dir,
                        sample_idx=idx
                    )

                    samples_processed += 1
                except Exception as sample_e:
                    logger.error(f"Error in evaluation sample {idx}: {sample_e}")
                    continue

        logger.info("Evaluation completed.")
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        raise


def save_model_weights(model: SamModel, save_path: str) -> None:
    """
    Save the model weights to the specified path.

    Args:
        model (SamModel): The trained SAM model.
        save_path (str): Directory to save the weights.
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, "sam_model.pt")
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model weights saved to {model_save_path}")
    except Exception as e:
        logger.error(f"Error saving model weights: {e}")
        raise


def save_training_plot(losses: List[float], save_dir: str = "plots") -> None:
    """
    Save the training loss plot using Plotly.

    Args:
        losses (List[float]): List of mean losses per epoch.
        save_dir (str, optional): Directory to save plots. Defaults to "plots".
    """
    try:
        os.makedirs(save_dir, exist_ok=True)

        epochs = list(range(1, len(losses) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=losses, mode='lines+markers', name='Training Loss'))
        fig.update_layout(
            title="Training Loss over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            width=800,
            height=600
        )

        plot_path = os.path.join(save_dir, "training_loss.html")
        save_plotly_figure(fig, plot_path)
    except Exception as e:
        logger.error(f"Error saving training plot: {e}")
        raise


def main() -> None:
    """Main function to execute the training pipeline."""
    try:
        set_seed()

        # Directories
        plots_dir = "plots"
        weights_dir = "model_weights"

        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset("hf-vision/road-pothole-segmentation")

        # Initialize processor and dataset
        logger.info("Initializing processor and datasets...")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        train_dataset = SAMDataset(dataset=dataset["train"], processor=processor)
        validation_dataset = SAMDataset(dataset=dataset["validation"], processor=processor)

        # Create DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

        # Initialize model
        logger.info("Loading model...")
        model = SamModel.from_pretrained("facebook/sam-vit-base")

        # Freeze parameters except mask decoder
        logger.info("Freezing vision and prompt encoders...")
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad = False

        # Define optimizer
        optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Using device: {device}")

        # Train the model
        logger.info("Starting training...")
        num_epochs = 10
        epoch_losses = train_model(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs
        )

        # Save training plot
        save_training_plot(epoch_losses, save_dir=plots_dir)

        # Save model weights
        save_model_weights(model, save_path=weights_dir)

        # Evaluate the model
        logger.info("Starting evaluation...")
        evaluate_model(
            model=model,
            dataset=validation_dataset,
            processor=processor,
            device=device,
            save_dir=plots_dir,
            num_samples=5
        )

        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()