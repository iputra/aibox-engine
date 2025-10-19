"""
Text chunking service for splitting documents into smaller pieces using LangChain text splitters.
"""

import logging
from typing import List, Optional, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunkingService:
    """Service for chunking text into smaller pieces for processing using LangChain text splitters."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize text chunking service.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size of a chunk in characters
            separators: List of separators for recursive splitting (default: LangChain defaults)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Default separators for recursive character splitting
        if separators is None:
            separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

        # Initialize LangChain text splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        logger.info(f"Initialized TextChunkingService with chunk_size={chunk_size}, "
                   f"overlap={chunk_overlap}")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using LangChain's RecursiveCharacterTextSplitter.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # For short texts, return as single chunk
        if len(text) <= self.chunk_size:
            return [text.strip()]

        try:
            # Use recursive chunking
            chunks = self._recursive_chunk(text)

            # Filter chunks by minimum size
            filtered_chunks = [
                chunk.strip() for chunk in chunks
                if len(chunk.strip()) >= self.min_chunk_size
            ]

            logger.info(f"Chunked text into {len(filtered_chunks)} chunks using recursive splitting")
            return filtered_chunks

        except Exception as e:
            logger.error(f"Error during chunking: {str(e)}")
            # Fallback to simple splitting
            return self._fallback_chunk(text)

    def _recursive_chunk(self, text: str) -> List[str]:
        """
        Perform recursive character splitting using LangChain's RecursiveCharacterTextSplitter.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        try:
            documents = self.recursive_splitter.create_documents([text])
            return [doc.page_content for doc in documents]
        except Exception as e:
            logger.error(f"Recursive chunking failed: {str(e)}")
            return self._fallback_chunk(text)

    def _fallback_chunk(self, text: str) -> List[str]:
        """
        Fallback chunking method that performs simple size-based splitting.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # Try to find a good breaking point
            chunk = text[start:end]
            space_pos = chunk.rfind(' ')
            if space_pos > self.chunk_size * 0.8:
                chunk = chunk[:space_pos]
                end = start + len(chunk)

            chunks.append(chunk.strip())
            start = max(start + 1, end - self.chunk_overlap)

        return [chunk for chunk in chunks if chunk]

    def chunk_text_with_metadata(self, text: str, document_id: int) -> List[Dict[str, Any]]:
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

    def get_chunk_stats(self, chunks: List[str]) -> Dict[str, Any]:
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
                "chunking_method": "recursive",
                "config": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "min_chunk_size": self.min_chunk_size,
                }
            }

        sizes = [len(chunk) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(sizes),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "chunking_method": "recursive",
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "min_chunk_size": self.min_chunk_size,
            }
        }

    def update_chunking_strategy(self,
                                chunk_size: Optional[int] = None,
                                chunk_overlap: Optional[int] = None,
                                separators: Optional[List[str]] = None) -> None:
        """
        Update the chunking strategy configuration.

        Args:
            chunk_size: New maximum chunk size
            chunk_overlap: New chunk overlap
            separators: New separators for recursive splitting
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap

        # Reinitialize the splitter with new configuration
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
        )

        logger.info(f"Updated chunking strategy: size={self.chunk_size}, "
                   f"overlap={self.chunk_overlap}")