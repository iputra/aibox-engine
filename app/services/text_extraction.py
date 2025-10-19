"""
Text extraction services for different document types using LangChain loaders.
"""

import logging
import os
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TextExtractionService:
    """Service for extracting text from various document types using LangChain loaders."""

    @staticmethod
    def _extract_with_langchain_loader(file_path: str, loader_class, **loader_kwargs) -> Optional[str]:
        """
        Extract text using a LangChain loader.

        Args:
            file_path: Path to the file
            loader_class: LangChain loader class to use
            **loader_kwargs: Additional arguments for the loader

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            loader = loader_class(file_path, **loader_kwargs)
            documents = loader.load()

            if not documents:
                logger.warning(f"No documents extracted from {file_path}")
                return None

            # Combine text from all pages/documents
            extracted_text = '\n\n'.join(doc.page_content for doc in documents)

            # Log extraction details
            if hasattr(loader, 'file_path'):
                file_type = os.path.splitext(loader.file_path)[1].lower()
            else:
                file_type = loader_class.__name__

            logger.info(f"Extracted {len(extracted_text)} characters from {file_type} file with {len(documents)} pages/sections")
            return extracted_text if extracted_text.strip() else None

        except Exception as e:
            logger.error(f"Error extracting text with {loader_class.__name__} from {file_path}: {str(e)}")
            return None

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Optional[str]:
        """
        Extract text from PDF file using LangChain's PyPDFLoader.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text or None if extraction fails
        """
        return TextExtractionService._extract_with_langchain_loader(
            file_path,
            PyPDFLoader
        )

    @staticmethod
    def extract_text_from_docx(file_path: str) -> Optional[str]:
        """
        Extract text from DOCX file using LangChain's Docx2txtLoader.

        Args:
            file_path: Path to DOCX file

        Returns:
            Extracted text or None if extraction fails
        """
        return TextExtractionService._extract_with_langchain_loader(
            file_path,
            Docx2txtLoader
        )

    @staticmethod
    def extract_text_from_txt(file_path: str) -> Optional[str]:
        """
        Extract text from TXT file using LangChain's TextLoader.

        Args:
            file_path: Path to TXT file

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            # Try with UTF-8 encoding first
            return TextExtractionService._extract_with_langchain_loader(
                file_path,
                TextLoader,
                encoding='utf-8'
            )
        except Exception as e:
            # Fallback to latin-1 encoding
            logger.warning(f"UTF-8 extraction failed, trying latin-1: {str(e)}")
            try:
                return TextExtractionService._extract_with_langchain_loader(
                    file_path,
                    TextLoader,
                    encoding='latin-1'
                )
            except Exception as e2:
                logger.error(f"Both encodings failed for TXT {file_path}: {str(e2)}")
                return None

    @staticmethod
    def extract_text_from_csv(file_path: str) -> Optional[str]:
        """
        Extract text from CSV file using LangChain's CSVLoader.

        Args:
            file_path: Path to CSV file

        Returns:
            Extracted text or None if extraction fails
        """
        return TextExtractionService._extract_with_langchain_loader(
            file_path,
            CSVLoader
        )

    @staticmethod
    def extract_text_from_markdown(file_path: str) -> Optional[str]:
        """
        Extract text from Markdown file using LangChain's UnstructuredMarkdownLoader.

        Args:
            file_path: Path to Markdown file

        Returns:
            Extracted text or None if extraction fails
        """
        return TextExtractionService._extract_with_langchain_loader(
            file_path,
            UnstructuredMarkdownLoader
        )

    @staticmethod
    def extract_text_from_html(file_path: str) -> Optional[str]:
        """
        Extract text from HTML file using LangChain's UnstructuredHTMLLoader.

        Args:
            file_path: Path to HTML file

        Returns:
            Extracted text or None if extraction fails
        """
        return TextExtractionService._extract_with_langchain_loader(
            file_path,
            UnstructuredHTMLLoader
        )

    @staticmethod
    def extract_text(file_path: str, file_type: str) -> Optional[str]:
        """
        Extract text from file based on its type using LangChain loaders.

        Args:
            file_path: Path to the file
            file_type: File type (extension)

        Returns:
            Extracted text or None if extraction fails
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        file_type = file_type.lower().lstrip('.')

        extraction_methods = {
            'pdf': TextExtractionService.extract_text_from_pdf,
            'docx': TextExtractionService.extract_text_from_docx,
            'txt': TextExtractionService.extract_text_from_txt,
            'csv': TextExtractionService.extract_text_from_csv,
            'md': TextExtractionService.extract_text_from_markdown,
            'markdown': TextExtractionService.extract_text_from_markdown,
            'html': TextExtractionService.extract_text_from_html,
            'htm': TextExtractionService.extract_text_from_html,
        }

        extractor = extraction_methods.get(file_type)
        if not extractor:
            logger.error(f"Unsupported file type: {file_type}")
            return None

        return extractor(file_path)

    @staticmethod
    def get_supported_file_types() -> list[str]:
        """
        Get list of supported file types.

        Returns:
            List of supported file extensions
        """
        return ['pdf', 'docx', 'txt', 'csv', 'md', 'markdown', 'html', 'htm']

    @staticmethod
    def is_supported_file_type(file_type: str) -> bool:
        """
        Check if file type is supported.

        Args:
            file_type: File extension

        Returns:
            True if supported, False otherwise
        """
        return file_type.lower().lstrip('.') in TextExtractionService.get_supported_file_types()