"""
Text chunking service for splitting documents into smaller pieces.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class TextChunkingService:
    """Service for chunking text into smaller pieces for processing."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        separator: str = "\n\n",
    ):
        """
        Initialize text chunking service.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size of a chunk in characters
            separator: Separator to use for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separator = separator

    def _split_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """
        Split text by a hierarchy of separators.

        Args:
            text: Text to split
            separators: List of separators in order of preference

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        current_chunks = [text]

        for separator in separators:
            new_chunks = []
            for chunk in current_chunks:
                if separator in chunk:
                    split_chunks = chunk.split(separator)
                    # Add back the separator to maintain context
                    for i, split_chunk in enumerate(split_chunks):
                        if i < len(split_chunks) - 1:
                            split_chunk += separator
                        if split_chunk.strip():
                            new_chunks.append(split_chunk.strip())
                else:
                    new_chunks.append(chunk)
            current_chunks = new_chunks

        return [chunk for chunk in current_chunks if chunk.strip()]

    def _recursive_character_split(self, text: str) -> List[str]:
        """
        Recursively split text by character to meet chunk size requirements.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk
                chunk = text[start:]
                if chunk.strip() and len(chunk) >= self.min_chunk_size:
                    chunks.append(chunk.strip())
                break

            # Try to find a good breaking point
            chunk = text[start:end]

            # Look for sentence boundaries
            sentence_end = chunk.rfind('. ')
            if sentence_end > self.chunk_size * 0.7:  # At least 70% of chunk size
                chunk = chunk[:sentence_end + 1]
                end = start + len(chunk)
            else:
                # Look for word boundaries
                space_pos = chunk.rfind(' ')
                if space_pos > self.chunk_size * 0.8:  # At least 80% of chunk size
                    chunk = chunk[:space_pos]
                    end = start + len(chunk)

            if chunk.strip() and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk.strip())

            # Calculate next start position with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using semantic and size-based strategies.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()

        if len(text) <= self.chunk_size:
            return [text]

        # First try semantic splitting using separators
        separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "? ",    # Question endings
            "! ",    # Exclamation endings
            "; ",    # Semicolons
            ", ",    # Commas
        ]

        semantic_chunks = self._split_by_separators(text, separators)

        # Further split chunks that are too large
        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk) <= self.chunk_size:
                if len(chunk) >= self.min_chunk_size:
                    final_chunks.append(chunk.strip())
            else:
                # Recursively split large chunks
                sub_chunks = self._recursive_character_split(chunk)
                final_chunks.extend(sub_chunks)

        logger.info(f"Chunked text into {len(final_chunks)} chunks")
        return final_chunks

    def chunk_text_with_metadata(self, text: str, document_id: int) -> List[dict]:
        """
        Chunk text and include metadata for each chunk.

        Args:
            text: Text to chunk
            document_id: ID of the source document

        Returns:
            List of chunks with metadata
        """
        chunks = self.chunk_text(text)

        chunk_with_metadata = []
        for index, chunk_text in enumerate(chunks):
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(chunk_text) // 4

            chunk_with_metadata.append({
                "document_id": document_id,
                "chunk_index": index,
                "content": chunk_text,
                "token_count": estimated_tokens,
            })

        return chunk_with_metadata

    def get_chunk_stats(self, chunks: List[str]) -> dict:
        """
        Get statistics about the chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        sizes = [len(chunk) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(sizes),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
        }