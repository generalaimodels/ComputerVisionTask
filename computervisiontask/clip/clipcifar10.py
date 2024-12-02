#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generalized Pipeline for Training, Evaluation, and Inference using CLIP on CIFAR-10.

This module implements a comprehensive pipeline that includes data loading, model training,
evaluation, inference, plotting results using Plotly, and saving model weights.

Author: Kandimalla Hemanth
Date: 12-2-2024
"""

import os
import sys
import logging
from typing import List, Tuple, Any, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Configuration and Constants
class Config:
    """Configuration parameters for the pipeline."""
    model_name: str = "openai/clip-vit-base-patch32"
    dataset_name: str = "cifar10"
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_dir: str = "plots"
    weights_dir: str = "model_weights"
    log_file: str = "pipeline.log"
    random_seed: int = 42

    @classmethod
    def initialize(cls):
        """Initialize directories and logging."""
        os.makedirs(cls.plot_dir, exist_ok=True)
        os.makedirs(cls.weights_dir, exist_ok=True)
        logging.basicConfig(
            filename=cls.log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        logging.info("Configuration initialized.")

# Data Handling
class CIFAR10DataModule:
    """Data module for CIFAR-10 dataset."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.labels = self._get_labels()

    def _get_labels(self) -> List[str]:
        """Retrieve labels from the dataset."""
        try:
            dataset = CIFAR10(root='./data', train=True, download=True)
            labels = dataset.classes
            logging.info(f"Labels retrieved: {labels}")
            return labels
        except Exception as e:
            logging.error(f"Error retrieving labels: {e}")
            raise

    def setup(self) -> None:
        """Set up dataset transforms."""
        try:
            self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            logging.info("Dataset transforms set up.")
        except Exception as e:
            logging.error(f"Error setting up transforms: {e}")
            raise

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        try:
            train_dataset = CIFAR10(
                root='./data',
                train=True,
                transform=self.train_transform,
                download=False
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2  # Reduced from 4 to 2
            )
            logging.info("Training data loader created.")
            return train_loader
        except Exception as e:
            logging.error(f"Error creating training data loader: {e}")
            raise

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        try:
            test_dataset = CIFAR10(
                root='./data',
                train=False,
                transform=self.test_transform,
                download=False
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2  # Reduced from 4 to 2
            )
            logging.info("Test data loader created.")
            return test_loader
        except Exception as e:
            logging.error(f"Error creating test data loader: {e}")
            raise

# Model Definition
class CLIPClassifier(nn.Module):
    """CLIP-based classifier for CIFAR-10."""

    def __init__(self, config: Config, labels: List[str]) -> None:
        super(CLIPClassifier, self).__init__()
        self.config = config
        self.labels = labels
        try:
            self.processor = AutoProcessor.from_pretrained(self.config.model_name)
            self.model = CLIPModel.from_pretrained(self.config.model_name).to(self.config.device)
            self.text_embeddings = self._prepare_text_embeddings()
            logging.info("CLIP model and text embeddings initialized.")
        except Exception as e:
            logging.error(f"Error initializing CLIP model: {e}")
            raise

    def _prepare_text_embeddings(self) -> torch.Tensor:
        """Prepare text embeddings for labels."""
        try:
            texts = [f"a photo of a {label}" for label in self.labels]
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.config.device)
            with torch.no_grad():
                text_outputs = self.model.get_text_features(**inputs)
                text_embeddings = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
            logging.info("Text embeddings prepared.")
            return text_embeddings
        except Exception as e:
            logging.error(f"Error preparing text embeddings: {e}")
            raise

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute image embeddings."""
        try:
            image_outputs = self.model.get_image_features(pixel_values=images)  # Updated argument
            image_embeddings = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)
            logits = image_embeddings @ self.text_embeddings.t()
            return logits
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            raise

# Plotting
class Plotter:
    """Handles plotting and saving of metrics and confusion matrices."""

    @staticmethod
    def plot_metrics(
        epochs: List[int],
        accuracies: List[float],
        precisions: List[float],
        recalls: List[float],
        f1_scores: List[float],
        save_path: str
    ) -> None:
        """Plot and save training metrics."""
        try:
            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=("Accuracy", "Precision", "Recall", "F1 Score"))
            fig.add_trace(go.Scatter(x=epochs, y=accuracies, mode='lines+markers', name='Accuracy'), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=precisions, mode='lines+markers', name='Precision'), row=1, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=recalls, mode='lines+markers', name='Recall'), row=2, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=f1_scores, mode='lines+markers', name='F1 Score'), row=2, col=2)
            fig.update_layout(title_text="Training Metrics", height=800)
            fig.write_html(save_path)
            logging.info(f"Metrics plot saved to {save_path}.")
        except Exception as e:
            logging.error(f"Error plotting metrics: {e}")
            raise

    @staticmethod
    def plot_confusion_matrix(
        cm: Any,
        labels: List[str],
        save_path: str
    ) -> None:
        """Plot and save confusion matrix."""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Viridis',
                reversescale=True,
                showscale=True
            ))
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='True Label',
                width=800,
                height=800
            )
            fig.write_html(save_path)
            logging.info(f"Confusion matrix plot saved to {save_path}.")
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")
            raise

# Utilities
def save_model(model: nn.Module, path: str) -> None:
    """Save the model weights to the specified path."""
    try:
        torch.save(model.state_dict(), path)
        logging.info(f"Model weights saved to {path}.")
    except Exception as e:
        logging.error(f"Error saving model weights: {e}")
        raise

# Training and Evaluation
def train(
    model: CLIPClassifier,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    config: Config,
    epoch: int
) -> float:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    try:
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            images, labels = batch
            images = images.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        logging.info(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        return epoch_acc
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

def evaluate(
    model: CLIPClassifier,
    dataloader: DataLoader,
    config: Config,
    epoch: Optional[int] = None
) -> Tuple[float, float, float, float, Any]:
    """Evaluate the model."""
    model.eval()
    y_true = []
    y_pred = []

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images, labels = batch
                images = images.to(config.device)
                labels = labels.to(config.device)

                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        if epoch is not None:
            logging.info(
                f"Epoch {epoch+1} - Eval Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1 Score: {f1:.4f}"
            )
        else:
            logging.info(
                f"Eval Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1 Score: {f1:.4f}"
            )

        return accuracy, precision, recall, f1, conf_matrix
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

# Main Execution
def main() -> None:
    """Main function to execute the pipeline."""
    try:
        # Initialize configuration
        Config.initialize()
        config = Config()

        # Setup data
        data_module = CIFAR10DataModule(config)
        data_module.setup()
        train_loader = data_module.train_dataloader()
        test_loader = data_module.test_dataloader()

        # Initialize model
        model = CLIPClassifier(config, data_module.labels)

        # Define optimizer and loss criterion
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Lists to store metrics
        epochs = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        # Training loop
        for epoch in range(config.num_epochs):
            epoch_acc = train(model, train_loader, optimizer, criterion, config, epoch)
            acc, prec, rec, f1, _ = evaluate(model, test_loader, config, epoch)
            epochs.append(epoch + 1)
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)

            # Save model weights
            model_path = os.path.join(config.weights_dir, f"clip_cifar10_epoch_{epoch+1}.pt")
            save_model(model, model_path)

        # Plot metrics
        metrics_plot_path = os.path.join(config.plot_dir, "training_metrics.html")
        Plotter.plot_metrics(epochs, accuracies, precisions, recalls, f1_scores, metrics_plot_path)

        # Final evaluation and confusion matrix
        final_acc, final_prec, final_rec, final_f1, conf_matrix = evaluate(model, test_loader, config)
        cm_plot_path = os.path.join(config.plot_dir, "confusion_matrix.html")
        Plotter.plot_confusion_matrix(conf_matrix, data_module.labels, cm_plot_path)

        # Save final model weights
        final_model_path = os.path.join(config.weights_dir, "clip_cifar10_final.pt")
        save_model(model, final_model_path)

        logging.info("Pipeline execution completed successfully.")

    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()