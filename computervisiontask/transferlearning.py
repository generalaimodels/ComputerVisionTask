#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIFAR-10 Classification using Pre-trained VGG19 Model.

This script trains and validates a VGG19 model on the CIFAR-10 dataset with enhanced
performance, scalability, and robustness. It includes comprehensive error handling,
type hints, and adheres to PEP-8 standards for readability and maintainability.
"""

import os
import sys
import time
import copy
import logging
from typing import Tuple, Dict, Optional

import numpy as np
import collections
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import datasets, models
from torchvision.transforms import Compose, Resize, AutoAugment, AutoAugmentPolicy, ToTensor, Normalize, v2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CIFAR10Trainer:
    """
    A class to handle the training and validation of a VGG19 model on the CIFAR-10 dataset.
    """

    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        num_classes: int = 10,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
        std: Tuple[float, float, float] = (0.247, 0.243, 0.261),
        model_path: str = 'pre_vgg19.pt',
    ) -> None:
        """
        Initialize the CIFAR10Trainer with configuration parameters.

        Args:
            data_dir (str): Directory to download/load the CIFAR-10 dataset.
            batch_size (int): Number of samples per batch.
            learning_rate (float): Initial learning rate for the optimizer.
            num_epochs (int): Number of training epochs.
            num_classes (int): Number of classes in the dataset.
            image_size (int): Size to which images are resized.
            mean (Tuple[float, float, float]): Mean for normalization.
            std (Tuple[float, float, float]): Standard deviation for normalization.
            model_path (str): Path to save the best model weights.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.model_path = model_path

        self.device = self._get_device()
        self._configure_cudnn()

        self.train_transforms = self._get_transforms(train=True)
        self.validation_transforms = self._get_transforms(train=False)

        self.train_set = self._get_dataset(train=True)
        self.validation_set = self._get_dataset(train=False)

        self.train_loader, self.validation_loader = self._get_dataloaders()

        self.model = self._initialize_model()

        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-6)

        self.loss_history: Dict[str, list] = {"train": [], "val": []}
        self.metric_history: Dict[str, list] = {"train": [], "val": []}

    def _get_device(self) -> torch.device:
        """
        Select the available device (GPU if available, else CPU).

        Returns:
            torch.device: The selected device.
        """
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f'Using device: {device}')
            return device
        except Exception as e:
            logger.error(f'Error selecting device: {e}')
            sys.exit(1)

    def _configure_cudnn(self) -> None:
        """
        Configure CuDNN for optimized performance.
        """
        try:
            cudnn.benchmark = True
            logger.info('CuDNN benchmark mode enabled.')
        except Exception as e:
            logger.warning(f'CuDNN configuration failed: {e}')

    def _get_transforms(self, train: bool = True) -> Compose:
        """
        Get the data transformations for training or validation.

        Args:
            train (bool): Flag indicating whether to get training transforms.

        Returns:
            Compose: Composed transformations.
        """
        try:
            if train:
                transformations = Compose([
                    Resize((self.image_size, self.image_size)),
                    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                    ToTensor(),
                    Normalize(self.mean, self.std)
                ])
            else:
                transformations = Compose([
                    Resize((self.image_size, self.image_size)),
                    ToTensor(),
                    Normalize(self.mean, self.std)
                ])
            logger.info(f'{"Training" if train else "Validation"} transforms created.')
            return transformations
        except Exception as e:
            logger.error(f'Error creating transforms: {e}')
            sys.exit(1)

    def _get_dataset(self, train: bool = True) -> Dataset:
        """
        Get the CIFAR-10 dataset for training or validation.

        Args:
            train (bool): Flag indicating whether to get the training dataset.

        Returns:
            Dataset: CIFAR-10 dataset.
        """
        try:
            dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=train,
                download=True,
                transform=self.train_transforms if train else self.validation_transforms
            )
            logger.info(f'{"Training" if train else "Validation"} dataset loaded.')
            return dataset
        except Exception as e:
            logger.error(f'Error loading dataset: {e}')
            sys.exit(1)

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for training and validation datasets.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
        """
        try:
            num_gpus = torch.cuda.device_count()
            logger.info(f'Number of GPUs available: {num_gpus}')

            train_loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2 * num_gpus,
                pin_memory=True
            )

            validation_loader = DataLoader(
                self.validation_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2 * num_gpus,
                pin_memory=True
            )

            logger.info('DataLoaders created.')
            return train_loader, validation_loader
        except Exception as e:
            logger.error(f'Error creating DataLoaders: {e}')
            sys.exit(1)

    def _initialize_model(self) -> nn.Module:
        """
        Initialize the VGG19 model with pre-trained weights and modify the classifier.

        Returns:
            nn.Module: The modified VGG19 model.
        """
        try:
            model = models.vgg19(weights='IMAGENET1K_V1')
            model.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_classes)
            for param in model.features.parameters():
                param.requires_grad = False

            model = model.to(self.device)
            logger.info('Model initialized and moved to device.')
            return model
        except Exception as e:
            logger.error(f'Error initializing model: {e}')
            sys.exit(1)

    @staticmethod
    def plot_class_distribution(dataset: Dataset, dataset_name: str) -> None:
        """
        Print and plot the class distribution of a dataset.

        Args:
            dataset (Dataset): The dataset to analyze.
            dataset_name (str): The name of the dataset.
        """
        try:
            labels = [label for _, label in dataset]
            counter = collections.Counter(labels)
            logger.info(f"Class Image Counter for {dataset_name} Data: {counter}")

            plt.figure(figsize=(10, 6))
            plt.bar(counter.keys(), counter.values(), color='skyblue')
            plt.xlabel("Class")
            plt.ylabel("Number of Images")
            plt.title(f"Class Distribution for {dataset_name} Data")
            plt.xticks(ticks=range(10), labels=dataset.classes, rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting class distribution: {e}')

    @staticmethod
    def imshow(tensor: torch.Tensor, title: Optional[str] = None) -> None:
        """
        Display a batch of images in a grid.

        Args:
            tensor (torch.Tensor): The input tensor containing the images.
            title (Optional[str]): The title of the plot.
        """
        try:
            image = torchvision.utils.make_grid(tensor).cpu().numpy().transpose((1, 2, 0))
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.247, 0.243, 0.261])
            image = std * image + mean
            image = np.clip(image, 0, 1)

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            if title:
                plt.title(title)
            plt.axis('off')
            plt.show()
        except Exception as e:
            logger.error(f'Error displaying images: {e}')

    @staticmethod
    def get_lr(optimizer: optim.Optimizer) -> float:
        """
        Get the current learning rate from the optimizer.

        Args:
            optimizer (optim.Optimizer): The optimizer object.

        Returns:
            float: The current learning rate.
        """
        try:
            lr = optimizer.param_groups[0]['lr']
            return lr
        except Exception as e:
            logger.error(f'Error retrieving learning rate: {e}')
            return 0.0

    @staticmethod
    def metrics_batch(output: torch.Tensor, target: torch.Tensor) -> int:
        """
        Count the number of correct predictions in a batch.

        Args:
            output (torch.Tensor): Model predictions.
            target (torch.Tensor): Target labels.

        Returns:
            int: Number of correct predictions.
        """
        try:
            pred = output.argmax(dim=1, keepdim=True)
            corrects = pred.eq(target.view_as(pred)).sum().item()
            return corrects
        except Exception as e:
            logger.error(f'Error computing batch metrics: {e}')
            return 0

    def loss_batch(
        self,
        loss_func: nn.Module,
        model_output: torch.Tensor,
        target: torch.Tensor,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Tuple[float, int]:
        """
        Compute the loss and metric for a batch of data.

        Args:
            loss_func (nn.Module): Loss function.
            model_output (torch.Tensor): Model predictions.
            target (torch.Tensor): Target labels.
            optimizer (Optional[optim.Optimizer]): Optimizer for backpropagation.

        Returns:
            Tuple[float, int]: Loss value and number of correct predictions.
        """
        try:
            loss_value = loss_func(model_output, target)
            corrects = self.metrics_batch(model_output, target)

            if optimizer:
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

            return loss_value.item(), corrects
        except Exception as e:
            logger.error(f'Error in loss_batch: {e}')
            return 0.0, 0

    def loss_epoch(
        self,
        model: nn.Module,
        loss_func: nn.Module,
        data_loader: DataLoader,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Tuple[float, float]:
        """
        Compute the average loss and metric over an epoch.

        Args:
            model (nn.Module): The neural network model.
            loss_func (nn.Module): Loss function.
            data_loader (DataLoader): DataLoader for the dataset.
            optimizer (Optional[optim.Optimizer]): Optimizer for backpropagation.

        Returns:
            Tuple[float, float]: Average loss and average accuracy.
        """
        model.train() if optimizer else model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            model_output = model(inputs)
            loss, corrects = self.loss_batch(loss_func, model_output, labels, optimizer)

            running_loss += loss
            running_corrects += corrects
            total_samples += labels.size(0)

        average_loss = running_loss / total_samples if total_samples else 0.0
        average_accuracy = running_corrects / total_samples if total_samples else 0.0

        return average_loss, average_accuracy

    def train_val(
        self,
        verbose: bool = False
    ) -> Tuple[nn.Module, Dict[str, list], Dict[str, list]]:
        """
        Train and validate the model.

        Args:
            verbose (bool): Whether to print detailed logs.

        Returns:
            Tuple[nn.Module, Dict[str, list], Dict[str, list]]: Trained model, loss history, metric history.
        """
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')

        for epoch in range(1, self.num_epochs + 1):
            current_lr = self.get_lr(self.optimizer)
            if verbose:
                logger.info(f'Epoch {epoch}/{self.num_epochs}, Current LR: {current_lr:.6f}')

            # Training phase
            train_loss, train_accuracy = self.loss_epoch(
                self.model, self.loss_func, self.train_loader, self.optimizer
            )
            self.loss_history['train'].append(train_loss)
            self.metric_history['train'].append(train_accuracy)

            # Validation phase
            val_loss, val_accuracy = self.loss_epoch(
                self.model, self.loss_func, self.validation_loader
            )
            self.loss_history['val'].append(val_loss)
            self.metric_history['val'].append(val_accuracy)

            # Check for improvement
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_wts, self.model_path)
                if verbose:
                    logger.info('Better model found and saved.')

            # Step the learning rate scheduler
            self.lr_scheduler.step()

            if verbose:
                logger.info(
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy * 100:.2f}% | '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%'
                )

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        logger.info('Training complete. Best model loaded.')

        return self.model, self.loss_history, self.metric_history

    @staticmethod
    def plot_history(
        loss_history: Dict[str, list],
        metric_history: Dict[str, list],
        epochs: Optional[int] = None
    ) -> None:
        """
        Plot training and validation loss and accuracy over epochs.

        Args:
            loss_history (Dict[str, list]): Dictionary containing 'train' and 'val' loss history.
            metric_history (Dict[str, list]): Dictionary containing 'train' and 'val' accuracy history.
            epochs (Optional[int]): Total number of epochs.
        """
        try:
            epochs = epochs or len(loss_history['train'])
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Model Loss", "Model Accuracy"]
            )

            # Plot Loss
            for phase, color in zip(['train', 'val'], ['#F1C40F', '#232323']):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, epochs + 1)),
                        y=loss_history[phase],
                        name=phase.capitalize(),
                        mode='lines',
                        line=dict(color=color)
                    ),
                    row=1, col=1
                )

            # Plot Accuracy
            for phase, color in zip(['train', 'val'], ['#F1C40F', '#232323']):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, epochs + 1)),
                        y=metric_history[phase],
                        name=phase.capitalize(),
                        mode='lines',
                        line=dict(color=color)
                    ),
                    row=1, col=2
                )

            fig.update_layout(
                template='plotly_white',
                title='Training and Validation Metrics',
                showlegend=True,
                height=500
            )
            fig.update_yaxes(range=[0, max(max(loss_history['train']), max(loss_history['val'])) * 1.1], row=1, col=1)
            fig.update_yaxes(range=[0, 1], row=1, col=2)  # Accuracy is between 0 and 1

            fig.show()
        except Exception as e:
            logger.error(f'Error plotting history: {e}')


def main() -> None:
    """
    Main function to execute the training and validation process.
    """
    try:
        # Configuration parameters
        config = {
            'data_dir': './data',
            'batch_size': 128,
            'learning_rate': 1e-3,
            'num_epochs': 10,
            'num_classes': 10,
            'image_size': 224,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.247, 0.243, 0.261),
            'model_path': 'pre_vgg19.pt'
        }

        trainer = CIFAR10Trainer(**config)

        # Optional: Plot class distribution
        trainer.plot_class_distribution(trainer.train_set, 'Training')

        # Train and validate the model
        start_time = time.time()
        model, loss_history, metric_history = trainer.train_val(verbose=True)
        end_time = time.time()

        logger.info(f'Training and validation completed in {end_time - start_time:.2f} seconds.')

        # Plot training history
        trainer.plot_history(loss_history, metric_history, epochs=config['num_epochs'])

    except Exception as e:
        logger.error(f'An error occurred in the main execution: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()