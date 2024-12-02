import os
from typing import List, Tuple, Any

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from datasets import load_dataset, Dataset
import faiss
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import tempfile
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPSimilaritySearch:
    """
    A class to perform prompt-based image similarity search using CLIP and FAISS,
    with results visualized using Plotly.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        dataset_name: str = "cifar10",
        dataset_split: str = "train",
        max_images: int = 10000,
        index_path: str = "clip.index",
        device: str = None,
    ) -> None:
        """
        Initialize the CLIPSimilaritySearch pipeline.

        Args:
            model_name (str): Name of the pre-trained CLIP model.
            dataset_name (str): Name of the dataset to load.
            dataset_split (str): Which split of the dataset to use.
            max_images (int): Maximum number of images to process.
            index_path (str): Path to save/load the FAISS index.
            device (str, optional): Device to run the model on. Defaults to CUDA if available.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.max_images = max_images
        self.index_path = index_path
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._load_model()
        self._load_tokenizer()
        self._load_dataset()
        self.embedding_dim = self._get_embedding_dim()
        self._initialize_faiss_index()

    def _load_model(self) -> None:
        """Load the CLIP model and processor."""
        try:
            logger.info(f"Loading processor and model '{self.model_name}'...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("Model and processor loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model '{self.model_name}': {e}")
            raise RuntimeError(f"Error loading model '{self.model_name}': {e}")

    def _load_tokenizer(self) -> None:
        """Load the tokenizer associated with the CLIP model."""
        try:
            logger.info(f"Loading tokenizer for '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading tokenizer for '{self.model_name}': {e}")
            raise RuntimeError(f"Error loading tokenizer for '{self.model_name}': {e}")

    def _load_dataset(self) -> None:
        """Load and preprocess the dataset."""
        try:
            logger.info(f"Loading dataset '{self.dataset_name}' split '{self.dataset_split}'...")
            dataset = load_dataset(self.dataset_name, split=self.dataset_split)
            if isinstance(dataset, Dataset):
                self.train_dataset: Dataset = dataset.select(range(min(self.max_images, len(dataset))))
            else:
                raise ValueError("Loaded dataset is not a HuggingFace Dataset object.")
            if len(self.train_dataset) == 0:
                raise ValueError("The filtered dataset is empty.")
            logger.info(f"Dataset loaded successfully with {len(self.train_dataset)} samples.")
        except Exception as e:
            logger.error(f"Error loading dataset '{self.dataset_name}': {e}")
            raise RuntimeError(f"Error loading dataset '{self.dataset_name}': {e}")

    def _get_embedding_dim(self) -> int:
        """Determine the embedding dimension from the model."""
        try:
            logger.info("Determining embedding dimension from the model...")
            with torch.no_grad():
                dummy_inputs = self.processor(images=Image.new("RGB", (224, 224)), return_tensors="pt")
                dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
                image_features = self.model.get_image_features(**dummy_inputs)
                embedding_dim = image_features.shape[1]
            logger.info(f"Embedding dimension determined: {embedding_dim}")
            return embedding_dim
        except Exception as e:
            logger.error(f"Error determining embedding dimension: {e}")
            raise RuntimeError(f"Error determining embedding dimension: {e}")

    def _initialize_faiss_index(self) -> None:
        """Initialize or load the FAISS index."""
        try:
            if os.path.exists(self.index_path):
                logger.info(f"Loading FAISS index from '{self.index_path}'...")
                self.index = faiss.read_index(self.index_path)
                if self.index.d != self.embedding_dim:
                    raise ValueError("Dimension mismatch in FAISS index.")
                logger.info("FAISS index loaded successfully.")
            else:
                logger.info("Initializing a new FAISS index.")
                # Use Inner Product for cosine similarity
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                logger.info("Initialized a new FAISS index with Inner Product.")
        except Exception as e:
            logger.error(f"Error initializing/loading FAISS index: {e}")
            # If loading fails, initialize a new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("Initialized a new FAISS index after failing to load existing index.")

    def _add_vectors_to_index(self, embeddings: np.ndarray) -> None:
        """Add normalized embeddings to the FAISS index."""
        try:
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            logger.info(f"Added {embeddings.shape[0]} vectors to the FAISS index.")
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS index: {e}")
            raise RuntimeError(f"Error adding vectors to FAISS index: {e}")

    def _extract_image_features(self, images: List[Any]) -> np.ndarray:
        """Extract image features using CLIP."""
        try:
            processed = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**processed)
            image_features = image_features.cpu().numpy().astype(np.float32)
            return image_features
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            raise RuntimeError(f"Error extracting image features: {e}")

    def build_index(self, batch_size: int = 32) -> None:
        """
        Build the FAISS index by processing the dataset in batches.

        Args:
            batch_size (int): Number of images to process in each batch.
        """
        try:
            logger.info("Starting to build the FAISS index...")
            total = len(self.train_dataset)
            for start_idx in tqdm(range(0, total, batch_size), desc="Building FAISS Index"):
                end_idx = min(start_idx + batch_size, total)
                batch = self.train_dataset[start_idx:end_idx]
                batch_images = [Image.fromarray(np.array(img)) for img in batch["img"]]
                embeddings = self._extract_image_features(batch_images)
                self._add_vectors_to_index(embeddings)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"FAISS index built and saved to '{self.index_path}'.")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise RuntimeError(f"Error building FAISS index: {e}")

    def search(
        self, prompt: str, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for images similar to the given prompt.

        Args:
            prompt (str): The text prompt to search for.
            top_k (int): Number of top similar images to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Similarities and indices of the top_k results.
        """
        try:
            logger.info(f"Embedding search prompt: '{prompt}'")
            text_inputs = self.tokenizer(
                [prompt], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(text_features)
            similarities, indices = self.index.search(text_features, top_k)
            logger.info("Search completed successfully.")
            return similarities, indices
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise RuntimeError(f"Error during search: {e}")

    def visualize_results(
        self, similarities: np.ndarray, indices: np.ndarray, prompt: str, output_html: str
    ) -> None:
        """
        Visualize the search results using Plotly and save as an HTML file.

        Args:
            similarities (np.ndarray): Similarity scores returned by FAISS.
            indices (np.ndarray): Indices of the top_k results.
            prompt (str): The original search prompt.
            output_html (str): Path to save the HTML visualization.
        """
        try:
            logger.info("Starting visualization of results...")
            fig = go.Figure()
            for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
                # Convert similarity from inner product to a percentage score
                similarity = sim * 100  # Since vectors are normalized, similarity ranges from -100 to 100
                image = self.train_dataset[int(idx)]["img"]
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    image.save(tmp_file.name)
                    fig.add_layout_image(
                        dict(
                            source=tmp_file.name,
                            xref="x",
                            yref="y",
                            x=(rank - 1) % 5,
                            y=((rank - 1) // 5),
                            sizex=1,
                            sizey=1,
                            sizing="stretch",
                            layer="below",
                        )
                    )
                fig.add_annotation(
                    x=(rank - 1) % 5,
                    y=((rank - 1) // 5),
                    text=f"Rank {rank}<br>Similarity: {similarity:.2f}%",
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                )
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig.update_layout(
                title=f"Top {len(indices[0])} Results for '{prompt}'",
                template="plotly_white",
                height=300 + 300 * ((len(indices[0]) - 1) // 5),
            )
            pio.write_html(fig, file=output_html, auto_open=False)
            logger.info(f"Visualization saved to '{output_html}'.")
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            raise RuntimeError(f"Error during visualization: {e}")

    def run_search_pipeline(
        self, prompt: str, top_k: int = 5, output_html: str = "results.html"
    ) -> None:
        """
        Execute the complete search pipeline: build index, perform search, and visualize.

        Args:
            prompt (str): The text prompt to search for.
            top_k (int, optional): Number of top results to retrieve. Defaults to 5.
            output_html (str, optional): Path to save the HTML visualization. Defaults to 'results.html'.
        """
        try:
            if self.index.ntotal == 0:
                logger.info("FAISS index is empty. Building index...")
                self.build_index()

            similarities, indices = self.search(prompt, top_k)
            for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
                print(f"Rank {rank}: Index {idx}, Similarity Score: {sim * 100:.2f}%")
            self.visualize_results(similarities, indices, prompt, output_html)
        except Exception as e:
            logger.error(f"An error occurred during the search pipeline: {e}")
            print(f"An error occurred: {e}")


def main() -> None:
    """
    Main function to execute the CLIP similarity search pipeline.
    """
    try:
        search_pipeline = CLIPSimilaritySearch(
            model_name="openai/clip-vit-base-patch32",
            dataset_name="cifar10",
            dataset_split="train",
            max_images=10000,
            index_path="clip.index",
        )
        search_prompt = "a photo of a dog"
        search_pipeline.run_search_pipeline(
            prompt=search_prompt, top_k=5, output_html="similarity_results.html"
        )
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()