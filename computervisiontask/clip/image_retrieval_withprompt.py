import logging
from typing import List, Tuple, Optional

import faiss
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class CLIPFAISSIndexer:
    """A class to handle CLIP feature extraction and FAISS indexing."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the CLIP model, processor, and tokenizer.

        Args:
            model_name (str): Pretrained model name.
            device (Optional[torch.device]): Device to run the model on.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            logging.info("Loading CLIP processor...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            logging.info("Loading CLIP model...")
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            logging.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise

        self.index = faiss.IndexFlatL2(self.model.config.projection_dim)
        if not self.index.is_trained:
            self.index.train(np.empty((0, self.model.config.projection_dim), dtype=np.float32))
        logging.info("FAISS index initialized.")

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract image features using CLIP.

        Args:
            image (Image.Image): PIL Image.

        Returns:
            np.ndarray: Normalized image feature vector.
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = image_features.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(image_features)
            return image_features
        except Exception as e:
            logging.error(f"Error extracting image features: {e}")
            raise

    def add_vector_to_index(self, embedding: np.ndarray) -> None:
        """
        Add a feature vector to the FAISS index.

        Args:
            embedding (np.ndarray): Feature vector.
        """
        try:
            self.index.add(embedding)
            logging.debug("Added vector to FAISS index.")
        except Exception as e:
            logging.error(f"Error adding vector to FAISS index: {e}")
            raise

    def extract_text_features(self, prompt: str) -> np.ndarray:
        """
        Extract text features using CLIP.

        Args:
            prompt (str): Text prompt.

        Returns:
            np.ndarray: Normalized text feature vector.
        """
        try:
            tokens = self.tokenizer([prompt], return_tensors="pt", truncation=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                text_features = self.model.get_text_features(**tokens)
            text_features = text_features.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(text_features)
            return text_features
        except Exception as e:
            logging.error(f"Error extracting text features: {e}")
            raise

    def search(
        self, query_features: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index for the top_k nearest neighbors.

        Args:
            query_features (np.ndarray): Query feature vector.
            top_k (int): Number of top results to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of top_k results.
        """
        try:
            distances, indices = self.index.search(query_features, top_k)
            logging.info(f"Search completed. Retrieved {top_k} results.")
            return distances, indices
        except Exception as e:
            logging.error(f"Error during FAISS search: {e}")
            raise

    def save_index(self, file_path: str) -> None:
        """
        Save the FAISS index to a file.

        Args:
            file_path (str): Path to save the index.
        """
        try:
            faiss.write_index(self.index, file_path)
            logging.info(f"FAISS index saved to {file_path}.")
        except Exception as e:
            logging.error(f"Error saving FAISS index: {e}")
            raise

    def load_index(self, file_path: str) -> None:
        """
        Load the FAISS index from a file.

        Args:
            file_path (str): Path to load the index from.
        """
        try:
            self.index = faiss.read_index(file_path)
            logging.info(f"FAISS index loaded from {file_path}.")
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            raise


def load_and_limit_dataset(
    dataset_name: str, subset: str, limit: int
) -> Dataset:
    """
    Load a dataset and limit its size.

    Args:
        dataset_name (str): Name of the dataset to load.
        subset (str): Subset to load (e.g., 'train', 'test').
        limit (int): Maximum number of samples.

    Returns:
        Dataset: A filtered Dataset object.
    """
    try:
        logging.info(f"Loading dataset '{dataset_name}'...")
        dataset_dict: DatasetDict = load_dataset(dataset_name)
        dataset: Dataset = dataset_dict[subset].select(range(limit))
        logging.info(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples.")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset '{dataset_name}': {e}")
        raise


def display_image(image: Image.Image, size: Tuple[int, int] = (200, 200)) -> None:
    """
    Display an image resized to the specified dimensions.

    Args:
        image (Image.Image): PIL Image.
        size (Tuple[int, int], optional): Desired size. Defaults to (200, 200).
    """
    try:
        resized_image = image.resize(size)
        resized_image.show()
    except Exception as e:
        logging.error(f"Error displaying image: {e}")


def compute_similarity_score(distance: float) -> float:
    """
    Compute similarity score from distance.

    Args:
        distance (float): Distance from FAISS search.

    Returns:
        float: Similarity score scaled between 0 and 1.
    """
    try:
        if distance == 0:
            return 100.0
        similarity = (1 / (1 + distance)) * 100
        return similarity
    except Exception as e:
        logging.error(f"Error computing similarity score: {e}")
        return 0.0


def main() -> None:
    """Main function to execute the CLIP-FAISS workflow."""
    try:
        # Configuration
        DATASET_NAME = "cifar10"
        SUBSET = "train"
        LIMIT = 10000
        INDEX_FILE = "clip.index"
        QUERY_PROMPT = "a photo of a dog"
        TOP_K = 5

        # Initialize indexer
        indexer = CLIPFAISSIndexer()

        # Load and prepare dataset
        dataset = load_and_limit_dataset(DATASET_NAME, SUBSET, LIMIT)

        # Process and index dataset images
        logging.info("Extracting and indexing image features...")
        for sample in tqdm(dataset, desc="Indexing images"):
            img: Image.Image = sample["img"]
            features = indexer.extract_image_features(img)
            indexer.add_vector_to_index(features)

        # Save the index for future use
        indexer.save_index(INDEX_FILE)

        # Perform a search query
        logging.info(f"Processing query: '{QUERY_PROMPT}'")
        text_features = indexer.extract_text_features(QUERY_PROMPT)
        distances, indices = indexer.search(text_features, top_k=TOP_K)

        # Display results
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            similarity = compute_similarity_score(distance)
            logging.info(f"Rank {rank}: Index={idx}, Similarity={similarity:.2f}%")
            image: Image.Image = dataset[idx]["img"]
            display_image(image)

    except Exception as e:
        logging.error(f"An error occurred in the main workflow: {e}")


if __name__ == "__main__":
    main()