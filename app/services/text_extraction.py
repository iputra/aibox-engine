"""
Text extraction services for different document types.
"""

import logging
import os
from typing import Optional

import docx
import PyPDF2

logger = logging.getLogger(__name__)


class TextExtractionService:
    """Service for extracting text from various document types."""

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Optional[str]:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            text = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():
                        text.append(page_text)

            extracted_text = '\n\n'.join(text)
            logger.info(f"Extracted {len(extracted_text)} characters from PDF with {num_pages} pages")
            return extracted_text if extracted_text.strip() else None

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return None

    @staticmethod
    def extract_text_from_docx(file_path: str) -> Optional[str]:
        """
        Extract text from DOCX file.

        Args:
            file_path: Path to DOCX file

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            doc = docx.Document(file_path)
            text = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)

            extracted_text = '\n\n'.join(text)
            logger.info(f"Extracted {len(extracted_text)} characters from DOCX with {len(text)} paragraphs")
            return extracted_text if extracted_text.strip() else None

        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            return None

    @staticmethod
    def extract_text_from_txt(file_path: str) -> Optional[str]:
        """
        Extract text from TXT file.

        Args:
            file_path: Path to TXT file

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            extracted_text = text.strip()
            logger.info(f"Extracted {len(extracted_text)} characters from TXT file")
            return extracted_text if extracted_text else None

        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                extracted_text = text.strip()
                logger.info(f"Extracted {len(extracted_text)} characters from TXT file (latin-1 encoding)")
                return extracted_text if extracted_text else None
            except Exception as e:
                logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
            return None

    @staticmethod
    def extract_text(file_path: str, file_type: str) -> Optional[str]:
        """
        Extract text from file based on its type.

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
        return ['pdf', 'docx', 'txt']

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