import logging
from typing import Any, List, Tuple

import faiss
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class CLIPImageIndexer:
    """A class to handle CLIP model feature extraction and FAISS indexing for image datasets."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        dataset_name: str = "zh-plus/tiny-imagenet",
        index_path: str = "clip.index",
        device: str = None,
    ) -> None:
        """
        Initialize the CLIPImageIndexer.

        Args:
            model_name (str): Name of the pre-trained CLIP model.
            dataset_name (str): Name of the dataset to load.
            index_path (str): Path to save/load the FAISS index.
            device (str, optional): Device to run the model on. Automatically detects if not provided.
        """
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.processor = self._load_processor(model_name)
        self.model = self._load_model(model_name)
        self.dataset = self._load_dataset(dataset_name)
        self.index = self._initialize_faiss_index(dimension=512)
        self.index_path = index_path

    def _load_processor(self, model_name: str) -> AutoProcessor:
        """Load the CLIP processor."""
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            logging.info("Processor loaded successfully.")
            return processor
        except Exception as e:
            logging.error(f"Failed to load processor: {e}")
            raise

    def _load_model(self, model_name: str) -> CLIPModel:
        """Load the CLIP model."""
        try:
            model = CLIPModel.from_pretrained(model_name).to(self.device)
            model.eval()
            logging.info("Model loaded and moved to device.")
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def _load_dataset(self, dataset_name: str) -> DatasetDict:
        """Load the specified dataset."""
        try:
            dataset = load_dataset(dataset_name)
            logging.info(f"Dataset '{dataset_name}' loaded successfully.")
            if "valid" not in dataset:
                raise ValueError("Dataset does not contain a 'valid' split.")
            return dataset["valid"]
        except Exception as e:
            logging.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise

    def _initialize_faiss_index(self, dimension: int) -> faiss.IndexFlatL2:
        """Initialize a FAISS index."""
        try:
            index = faiss.IndexFlatL2(dimension)
            logging.info("FAISS index initialized.")
            return index
        except Exception as e:
            logging.error(f"Failed to initialize FAISS index: {e}")
            raise

    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract CLIP features for a batch of images.

        Args:
            images (List[PIL.Image.Image]): Batch of images.

        Returns:
            np.ndarray: Extracted and normalized feature vectors.
        """
        try:
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            features = image_features.cpu().numpy().astype("float32")
            faiss.normalize_L2(features)
            logging.debug(f"Extracted features for {len(images)} images.")
            return features
        except Exception as e:
            logging.error(f"Failed to extract features: {e}")
            raise

    def build_index(self, batch_size: int = 32) -> None:
        """
        Build the FAISS index from the dataset.

        Args:
            batch_size (int): Number of images to process in each batch.
        """
        try:
            total_images = len(self.dataset)
            logging.info(f"Building index for {total_images} images with batch size {batch_size}.")

            for idx in tqdm(range(0, total_images, batch_size), desc="Indexing", unit="batch"):
                batch = [self.dataset[i]["image"] for i in range(idx, min(idx + batch_size, total_images))]
                features = self.extract_features(batch)
                self.index.add(features)

            faiss.write_index(self.index, self.index_path)
            logging.info(f"FAISS index built and saved to '{self.index_path}'.")
        except Exception as e:
            logging.error(f"Failed to build FAISS index: {e}")
            raise

    def load_index(self) -> None:
        """Load a pre-built FAISS index from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            logging.info(f"FAISS index loaded from '{self.index_path}'.")
        except Exception as e:
            logging.error(f"Failed to load FAISS index: {e}")
            raise

    def save_index(self) -> None:
        """Save the FAISS index to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            logging.info(f"FAISS index saved to '{self.index_path}'.")
        except Exception as e:
            logging.error(f"Failed to save FAISS index: {e}")
            raise

    def search_similar_images(self, input_image: Image.Image, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar images in the index given an input image.

        Args:
            input_image (PIL.Image.Image): The input image to search with.
            top_k (int): Number of top similar images to retrieve.

        Returns:
            List[Tuple[int, float]]: List of tuples containing indices and similarity scores.
        """
        try:
            features = self.extract_features([input_image])
            distances, indices = self.index.search(features, top_k)
            similarities = (1 / (1 + distances)) * 100  # Convert distances to a similarity score between 0 and 100
            results = list(zip(indices[0], similarities[0]))
            logging.info(f"Found top {top_k} similar images.")
            return results
        except Exception as e:
            logging.error(f"Failed to search similar images: {e}")
            raise

    def display_results(self, results: List[Tuple[int, float]]) -> None:
        """
        Display the search results.

        Args:
            results (List[Tuple[int, float]]): List of image indices and their similarity scores.
        """
        try:
            for idx, sim in results:
                image = self.dataset[int(idx)]["image"].resize((200, 200))
                print(f"Index: {idx}, Similarity Score: {sim:.2f}%")
                image.show()
        except Exception as e:
            logging.error(f"Failed to display results: {e}")
            raise


def main() -> None:
    """Main function to execute the image indexing and searching."""
    try:
        # Initialize the indexer
        indexer = CLIPImageIndexer()

        # Build the FAISS index (comment out if index already built)
        indexer.build_index(batch_size=64)

        # Alternatively, load an existing index
        # indexer.load_index()

        # Example: Load an input image for searching
        input_image_url = "https://example.com/path/to/your/image.jpg"
        try:
            response = requests.get(input_image_url)
            response.raise_for_status()
            input_image = Image.open(BytesIO(response.content)).convert("RGB")
            logging.info("Input image loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load input image: {e}")
            return

        # Search for similar images
        top_k = 5
        results = indexer.search_similar_images(input_image, top_k=top_k)

        # Display the results
        indexer.display_results(results)

    except Exception as e:
        logging.critical(f"An unrecoverable error occurred: {e}")


if __name__ == "__main__":
    main()