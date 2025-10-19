"""
Embedding service using LangChain with llama.cpp and Nomic embedding model.
"""

import logging
import os
from typing import List, Optional, Dict, Any

from decouple import config
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using LangChain with llama.cpp."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_gpu_layers: int = 0,
        embedding_dimension: int = 768,
    ):
        """
        Initialize embedding service.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context size
            n_batch: Batch size for processing
            n_gpu_layers: Number of layers to offload to GPU
            embedding_dimension: Dimension of the embedding vectors
        """
        self.model_path = model_path or config(
            "NOMIC_EMBED_MODEL_PATH",
            default="models/nomic-embed-text-v1.5.Q4_K_M.gguf",
        )
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.embedding_dim = embedding_dimension  # Nomic embed text v1.5 dimension

        self.embeddings: Optional[Embeddings] = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model using LangChain."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                logger.info("Please download nomic-embed-text-v1.5.GGUF and place it in the models directory")
                return

            logger.info(f"Loading embedding model from {self.model_path}")
            self.embeddings = LlamaCppEmbeddings(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.embeddings = None

    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded successfully.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.embeddings is not None

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text using LangChain.

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

            # Generate embedding using LangChain
            embedding = self.embeddings.embed_query(text)

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts using LangChain.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failed generations)
        """
        if not self.is_model_loaded():
            logger.error("Embedding model not loaded")
            return [None] * len(texts)

        try:
            # Filter out empty texts
            valid_texts = []
            valid_indices = []

            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i)

            if not valid_texts:
                logger.warning("No valid texts provided for embedding")
                return [None] * len(texts)

            # Generate embeddings using LangChain's batch processing
            embeddings = self.embeddings.embed_documents(valid_texts)

            # Map results back to original indices
            result = [None] * len(texts)
            for i, embedding in zip(valid_indices, embeddings):
                result[i] = embedding

            successful_embeddings = sum(1 for e in result if e is not None)
            logger.info(f"Generated {successful_embeddings}/{len(texts)} embeddings successfully")

            return result

        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            # Fallback to individual processing
            return [self.generate_embedding(text) for text in texts]

    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks using LangChain.

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

    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding service.

        Returns:
            Dictionary with embedding service information
        """
        return {
            "model_path": self.model_path,
            "model_loaded": self.is_model_loaded(),
            "embedding_dimension": self.embedding_dim,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "backend": "langchain",
            "provider": "llama.cpp",
        }

    def update_model_config(self,
                           n_ctx: Optional[int] = None,
                           n_batch: Optional[int] = None,
                           n_gpu_layers: Optional[int] = None) -> bool:
        """
        Update model configuration and reload the model.

        Args:
            n_ctx: New context size
            n_batch: New batch size
            n_gpu_layers: New GPU layers count

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Update configuration
            if n_ctx is not None:
                self.n_ctx = n_ctx
            if n_batch is not None:
                self.n_batch = n_batch
            if n_gpu_layers is not None:
                self.n_gpu_layers = n_gpu_layers

            # Reload the model with new configuration
            self._load_model()

            logger.info(f"Updated model configuration: ctx={self.n_ctx}, "
                       f"batch={self.n_batch}, gpu_layers={self.n_gpu_layers}")
            return self.is_model_loaded()

        except Exception as e:
            logger.error(f"Failed to update model configuration: {str(e)}")
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