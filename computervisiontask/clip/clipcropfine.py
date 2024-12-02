import os
import logging
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import (
    CLIPProcessor,
    CLIPModel,
    YolosImageProcessor,
    YolosForObjectDetection,
    get_scheduler,
)
import requests
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings

warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Constants
DETECTION_MODEL_NAME = "hustvl/yolos-tiny"
MULTIMODAL_MODEL_NAME = "openai/clip-vit-base-patch16"
MODEL_SAVE_DIR = "model_weights"
PLOTS_SAVE_DIR = "plots"

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)


class CustomImageDataset(Dataset):
    """
    Custom dataset for handling image data.
    """

    def __init__(self, image_urls: List[str], processor: Any, transform: Optional[Any] = None):
        """
        Initialize the dataset with image URLs.

        :param image_urls: List of image URLs.
        :param processor: Image processor.
        :param transform: Optional transformations.
        """
        self.image_urls = image_urls
        self.processor = processor
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_urls)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        url = self.image_urls[idx]
        try:
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            if self.transform:
                image = self.transform(image)
            inputs = self.processor(images=image, return_tensors="pt")
            return {"image": image, "inputs": inputs}
        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {e}")
            return {"image": None, "inputs": None}


def extract_list_images_detected(
    image: Image.Image, prob_threshold: float, det_processor: Any, det_model: Any
) -> Tuple[List[Image.Image], List[float]]:
    """
    Perform object detection on an image and extract regions of interest.

    Args:
        image (PIL.Image.Image): Input image.
        prob_threshold (float): Probability threshold for detections.
        det_processor: Detection image processor.
        det_model: Detection model.

    Returns:
        Tuple[List[PIL.Image.Image], List[float]]: List of cropped images and their scores.
    """
    try:
        inputs = det_processor(images=image, return_tensors="pt")
        outputs = det_model(**inputs)

        logits = outputs.logits
        probas = logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > prob_threshold
        outs = det_processor.post_process(outputs, torch.tensor(image.size[::-1]).unsqueeze(0))
        bboxes_scaled = outs[0]["boxes"][keep].detach().numpy()
        scores = outs[0]["scores"][keep].detach().numpy()

        images_list = []
        for bbox in bboxes_scaled:
            xmin, ymin, xmax, ymax = map(int, bbox)
            roi = image.crop((xmin, ymin, xmax, ymax))
            images_list.append(roi)

        return images_list, scores
    except Exception as e:
        logger.error(f"Error in extract_list_images_detected: {e}")
        return [], []


def extract_image_with_description(
    images_list: List[Image.Image], text: str, mm_processor: Any, mm_model: Any
) -> Tuple[Image.Image, float]:
    """
    Select the image that best matches the given text description.

    Args:
        images_list (List[PIL.Image.Image]): List of images.
        text (str): Text description.
        mm_processor: Multimodal processor.
        mm_model: Multimodal model.

    Returns:
        Tuple[PIL.Image.Image, float]: Best matching image and its score.
    """
    try:
        inputs = mm_processor(text=[text], images=images_list, return_tensors="pt", padding=True)
        outputs = mm_model(**inputs)
        logits_per_image = outputs.logits_per_text
        probs = logits_per_image.softmax(-1).detach().cpu().numpy()[0]
        best_idx = np.argmax(probs)
        return images_list[best_idx], float(probs[best_idx])
    except Exception as e:
        logger.error(f"Error in extract_image_with_description: {e}")
        return Image.new("RGB", (100, 100), color="red"), 0.0


def save_plotly_fig(fig: go.Figure, filename: str) -> None:
    """
    Save a Plotly figure as an HTML file.

    Args:
        fig (go.Figure): Plotly figure.
        filename (str): Filename to save the plot.
    """
    try:
        fig.write_html(os.path.join(PLOTS_SAVE_DIR, filename))
        logger.info(f"Plot saved as {filename}")
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {e}")


def save_model_weights(model: Any, save_path: str) -> None:
    """
    Save model weights to the specified path.

    Args:
        model: PyTorch model.
        save_path (str): Path to save the model weights.
    """
    try:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model weights saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving model weights to {save_path}: {e}")


def train_model(
    model: Any,
    dataloader: DataLoader,
    optimizer: Any,
    scheduler: Any,
    device: torch.device,
    num_epochs: int = 3,
) -> Tuple[List[float], List[float]]:
    """
    Train the model and track loss.

    Args:
        model: PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.

    Returns:
        Tuple[List[float], List[float]]: Training losses and learning rates.
    """
    model.to(device)
    loss_fn = CrossEntropyLoss()
    training_losses = []
    learning_rates = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch["inputs"]
            if inputs is None:
                continue
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            training_losses.append(loss.item())
            learning_rates.append(scheduler.get_last_lr()[0])

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return training_losses, learning_rates


def evaluate_model(
    model: Any, dataloader: DataLoader, device: torch.device
) -> Tuple[List[float], List[float]]:
    """
    Evaluate the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to evaluate on.

    Returns:
        Tuple[List[float], List[float]]: Evaluation losses and accuracies.
    """
    model.to(device)
    model.eval()
    loss_fn = CrossEntropyLoss()
    eval_losses = []
    accuracies = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"]
            if inputs is None:
                continue
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else loss_fn(outputs.logits, labels)
            eval_losses.append(loss.item())
            preds = outputs.logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item() if labels is not None else 0.0
            accuracies.append(acc)

    avg_loss = np.mean(eval_losses)
    avg_accuracy = np.mean(accuracies)
    logger.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return eval_losses, accuracies


def create_interactive_plot(
    training_losses: List[float],
    learning_rates: List[float],
    eval_losses: List[float],
    accuracies: List[float],
) -> go.Figure:
    """
    Create an interactive Plotly figure for training and evaluation metrics.

    Args:
        training_losses (List[float]): Training losses.
        learning_rates (List[float]): Learning rates.
        eval_losses (List[float]): Evaluation losses.
        accuracies (List[float]): Evaluation accuracies.

    Returns:
        go.Figure: Plotly figure.
    """
    try:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Training Metrics", "Evaluation Metrics"))

        fig.add_trace(
            go.Scatter(y=training_losses, mode="lines", name="Training Loss"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(y=learning_rates, mode="lines", name="Learning Rate"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(y=eval_losses, mode="lines", name="Validation Loss"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(y=accuracies, mode="lines", name="Validation Accuracy"),
            row=2,
            col=1,
        )

        fig.update_layout(height=800, width=1000, title_text="Training and Evaluation Metrics")
        return fig
    except Exception as e:
        logger.error(f"Error creating interactive plot: {e}")
        return go.Figure()


def main() -> None:
    """
    Main function to execute the pipeline.
    """
    try:
        # Initialize models and processors
        det_image_processor = YolosImageProcessor.from_pretrained(DETECTION_MODEL_NAME)
        det_model = YolosForObjectDetection.from_pretrained(DETECTION_MODEL_NAME)
        mm_model = CLIPModel.from_pretrained(MULTIMODAL_MODEL_NAME)
        mm_processor = CLIPProcessor.from_pretrained(MULTIMODAL_MODEL_NAME)

        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Example training and validation URLs (replace with actual dataset)
        train_image_urls = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            # Add more URLs
        ]
        val_image_urls = [
            "https://example.com/image3.jpg",
            "https://example.com/image4.jpg",
            # Add more URLs
        ]

        # Create datasets and dataloaders
        train_dataset = CustomImageDataset(train_image_urls, det_image_processor)
        val_dataset = CustomImageDataset(val_image_urls, det_image_processor)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        # Initialize optimizer and scheduler
        optimizer = AdamW(det_model.parameters(), lr=5e-5)
        num_epochs = 3
        num_training_steps = num_epochs * len(train_loader)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        # Train the detection model
        training_losses, learning_rates = train_model(
            det_model, train_loader, optimizer, scheduler, device, num_epochs
        )

        # Evaluate the detection model
        eval_losses, accuracies = evaluate_model(det_model, val_loader, device)

        # Create and save interactive plots
        fig = create_interactive_plot(training_losses, learning_rates, eval_losses, accuracies)
        save_plotly_fig(fig, "training_evaluation_metrics.html")

        # Save model weights
        save_model_weights(det_model, os.path.join(MODEL_SAVE_DIR, "yolos_model.pth"))
        save_model_weights(mm_model, os.path.join(MODEL_SAVE_DIR, "clip_model.pth"))

        # Inference example
        url = "https://i.pinimg.com/736x/1b/51/42/1b5142c269f2e9a356202af3f5569a87.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        list_rois, conf_scores = extract_list_images_detected(image, prob_threshold=0.85, det_processor=det_image_processor, det_model=det_model)

        TEXT_DESC = [
            "Man carrying a green bag",
            "Man riding a bicycle",
            "Yellow colored taxi car",
            "Red colored bus",
        ]
        images_list = []
        for txt in TEXT_DESC:
            img, score = extract_image_with_description(images_list=list_rois, text=txt, mm_processor=mm_processor, mm_model=mm_model)
            images_list.append({"image": img, "description": txt, "conf-score": score})

        # Create interactive plot for inference
        inference_fig = make_subplots(rows=1, cols=len(images_list), subplot_titles=[img["description"] for img in images_list])
        for idx, img_dict in enumerate(images_list, 1):
            img = img_dict["image"]
            score = img_dict["conf-score"]
            img_np = np.array(img)
            inference_fig.add_trace(
                go.Image(z=img_np),
                row=1,
                col=idx,
            )
            inference_fig.add_annotation(
                x=0.5,
                y=-0.15,
                xref=f"x{idx}",
                yref=f"y{idx}",
                text=f"Confidence Score: {score:.3f}",
                showarrow=False,
                font=dict(size=12),
            )
        inference_fig.update_layout(height=600, width=200 * len(images_list), title_text="Inference Results")
        save_plotly_fig(inference_fig, "inference_results.html")

    except Exception as e:
        logger.exception(f"An error occurred in the pipeline: {e}")


if __name__ == "__main__":
    main()