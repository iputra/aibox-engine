"""
Embedding service using llama.cpp with Nomic embedding model.
"""

import logging
import os
from typing import List, Optional

from decouple import config
from llama_cpp import Llama

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using llama.cpp."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_gpu_layers: int = 0,
    ):
        """
        Initialize embedding service.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context size
            n_batch: Batch size for processing
            n_gpu_layers: Number of layers to offload to GPU
        """
        self.model_path = model_path or config(
            "NOMIC_EMBED_MODEL_PATH",
            default="models/nomic-embed-text-v1.5.Q4_K_M.gguf",
        )
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.embedding_dim = 768  # Nomic embed text v1.5 dimension

        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                logger.info("Please download nomic-embed-text-v1.5.GGUF and place it in the models directory")
                return

            logger.info(f"Loading embedding model from {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_gpu_layers=self.n_gpu_layers,
                embedding=True,  # Enable embedding mode
                verbose=False,
            )
            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.model = None

    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded successfully.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if generation fails
        """
        if not self.is_model_loaded():
            logger.error("Embedding model not loaded")
            return None

        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        try:
            # Clean and prepare text
            text = text.strip()

            # Generate embedding
            embedding = self.model.embed(text)

            # Convert to list if needed
            if isinstance(embedding, list):
                return embedding
            else:
                return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failed generations)
        """
        if not self.is_model_loaded():
            logger.error("Embedding model not loaded")
            return [None] * len(texts)

        embeddings = []

        for i, text in enumerate(texts):
            if i % 10 == 0:
                logger.info(f"Processing embedding {i+1}/{len(texts)}")

            embedding = self.generate_embedding(text)
            embeddings.append(embedding)

        successful_embeddings = sum(1 for e in embeddings if e is not None)
        logger.info(f"Generated {successful_embeddings}/{len(texts)} embeddings successfully")

        return embeddings

    def generate_embeddings_for_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of chunk dictionaries with content and metadata

        Returns:
            List of chunks with embeddings added
        """
        if not self.is_model_loaded():
            logger.error("Embedding model not loaded")
            return []

        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.generate_embeddings_batch(texts)

        # Add embeddings to chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is not None:
                chunk_copy = chunk.copy()
                chunk_copy["embedding"] = embedding
                chunks_with_embeddings.append(chunk_copy)
            else:
                logger.warning(f"Failed to generate embedding for chunk {chunk.get('chunk_index', 'unknown')}")

        logger.info(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks")
        return chunks_with_embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim

    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate embedding vector.

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(embedding, list):
            return False

        if len(embedding) != self.embedding_dim:
            return False

        # Check if all values are numbers
        try:
            for value in embedding:
                float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine similarity score
        """
        try:
            import numpy as np

            a_np = np.array(a, dtype=np.float32)
            b_np = np.array(b, dtype=np.float32)

            dot_product = np.dot(a_np, b_np)
            norm_a = np.linalg.norm(a_np)
            norm_b = np.linalg.norm(b_np)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(dot_product / (norm_a * norm_b))

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0