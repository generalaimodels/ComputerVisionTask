import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, Dataset
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, ConfusionMatrixDisplay)
from tqdm import tqdm
from transformers import (AutoProcessor, AutoTokenizer, CLIPModel)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class CLIPClassifier:
    """
    A classifier using the CLIP model for image classification.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None) -> None:
        """
        Initializes the CLIPClassifier by loading the model, processor, and tokenizer.

        Args:
            model_name (str): The pretrained CLIP model name.
            device (str, optional): The device to run the model on. Defaults to CUDA if available.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")

        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded model '{model_name}' successfully.")
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            raise

    def classify_batch(self, images: List[Image.Image], texts: List[str], batch_size: int = 32) -> List[int]:
        """
        Classifies a batch of images against a list of text labels.

        Args:
            images (List[Image.Image]): List of PIL Images to classify.
            texts (List[str]): List of text labels.
            batch_size (int): Number of samples per batch for processing.

        Returns:
            List[int]: Predicted label indices for each image.
        """
        predictions = []
        labels_tensor = torch.arange(len(texts)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Classifying Batches"):
                batch_images = images[i:i + batch_size]
                try:
                    inputs = self.processor(text=texts, images=batch_images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image  # Shape: (batch_size, num_labels)
                    probs = logits_per_image.softmax(dim=1)
                    batch_preds = torch.argmax(probs, dim=1).cpu().tolist()
                    predictions.extend(batch_preds)
                except Exception as e:
                    logger.error(f"Error during batch classification: {e}")
                    predictions.extend([-1] * len(batch_images))  # Assign -1 for failed predictions
        return predictions


def compute_metrics(y_true: List[int], y_pred: List[int], labels: List[str]) -> None:
    """
    Computes and prints evaluation metrics.

    Args:
        y_true (List[int]): Ground truth labels.
        y_pred (List[int]): Predicted labels.
        labels (List[str]): List of label names.
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        logger.info(f'Accuracy: {accuracy:.4f}')
        logger.info(f'Precision: {precision:.4f}')
        logger.info(f'Recall: {recall:.4f}')
        logger.info(f'F1 Score: {f1:.4f}')

        logger.info('\nClassification Report:')
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        logger.info(f'{report}')

        conf_matrix = confusion_matrix(y_true, y_pred)
        logger.info('Confusion Matrix:')
        logger.info(f'{conf_matrix}')

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
        disp.plot(include_values=True, cmap='viridis', xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise


def main() -> None:
    """
    Main function to execute the classification and evaluation.
    """
    try:
        # Initialize classifier
        classifier = CLIPClassifier()

        # Load CIFAR-10 dataset
        dataset: Dataset = load_dataset("cifar10")
        labels: List[str] = dataset["train"].features["label"].names
        logger.info(f"Dataset labels: {labels}")

        # Prepare test data
        test_dataset = dataset['test']
        ground_truth: List[int] = test_dataset['label']
        test_images: List[Image.Image] = [item['img'] for item in test_dataset]

        # Initial classification with original labels
        logger.info("Starting classification with original labels...")
        predictions = classifier.classify_batch(test_images, labels)
        compute_metrics(ground_truth, predictions, labels)

        # Define new labels
        new_labels = [f"a photo of {label}" for label in labels]
        logger.info(f"New Labels: {new_labels}")

        # Classification with new labels
        logger.info("Starting classification with modified labels...")
        predictions_new = classifier.classify_batch(test_images, new_labels)
        compute_metrics(ground_truth, predictions_new, labels)  # Using original labels for display

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()