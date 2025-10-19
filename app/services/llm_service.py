"""
LLM service for AIBox Engine using OpenAI-compatible APIs.
"""

import logging
from typing import List, Optional, Dict, Any
from decouple import config
import httpx

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with LLM APIs using OpenAI-compatible format."""

    def __init__(self):
        """Initialize LLM service with configuration from environment variables."""
        self.api_key = config("LLM_API_KEY", default=None)
        self.base_url = config("LLM_BASE_URL", default=None)
        self.model = config("LLM_MODEL", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
        self.default_temperature = float(config("LLM_TEMPERATURE", default="0.7"))
        self.default_max_tokens = int(config("LLM_MAX_TOKENS", default="1000"))

        # Check if service is configured
        self.is_configured = bool(self.api_key and self.base_url)

        if not self.is_configured:
            logger.warning("LLM service not configured. Please set LLM_API_KEY and LLM_BASE_URL in environment variables.")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate response from LLM API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response

        Returns:
            Response dictionary or None if generation fails
        """
        if not self.is_configured:
            logger.error("LLM service not configured")
            return None

        try:
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.default_temperature,
                "max_tokens": max_tokens or self.default_max_tokens,
                "stream": stream,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Make API request
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"LLM API error: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return None

    async def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Generate streaming response from LLM API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Yields:
            Response chunks from the streaming API
        """
        if not self.is_configured:
            logger.error("LLM service not configured")
            return

        try:
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.default_temperature,
                "max_tokens": max_tokens or self.default_max_tokens,
                "stream": True,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Make streaming API request
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]  # Remove "data: " prefix
                                if data.strip() == "[DONE]":
                                    break
                                yield data
                    else:
                        logger.error(f"LLM API streaming error: {response.status_code}")
                        return

        except Exception as e:
            logger.error(f"Error generating LLM streaming response: {str(e)}")
            return

    def build_context_messages(
        self,
        user_message: str,
        document_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build message list with context for LLM.

        Args:
            user_message: Current user message
            document_context: Context from document search
            system_prompt: Custom system prompt
            conversation_history: Previous messages in conversation

        Returns:
            List of formatted messages for LLM API
        """
        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif document_context:
            messages.append({
                "role": "system",
                "content": (
                    "You are a helpful AI assistant with access to document context. "
                    "Use the provided document information to answer questions accurately. "
                    "Cite your sources when referencing specific information from the documents. "
                    "If the documents don't contain relevant information, say so politely."
                )
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant. Provide accurate and helpful responses."
            })

        # Add conversation history (last 10 messages to maintain context)
        if conversation_history:
            messages.extend(conversation_history[-10:])

        # Add document context if available
        if document_context:
            messages.append({
                "role": "system",
                "content": f"Document Context:\n{document_context}"
            })

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def extract_response_text(self, response_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract text content from LLM response.

        Args:
            response_data: Raw response from LLM API

        Returns:
            Response text or None if extraction fails
        """
        try:
            if response_data and "choices" in response_data:
                choice = response_data["choices"][0]
                if "message" in choice:
                    return choice["message"]["content"]
            return None
        except Exception as e:
            logger.error(f"Error extracting response text: {str(e)}")
            return None

    def extract_usage_info(self, response_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract usage information from LLM response.

        Args:
            response_data: Raw response from LLM API

        Returns:
            Dictionary with usage metrics
        """
        try:
            if response_data and "usage" in response_data:
                usage = response_data["usage"]
                return {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except Exception as e:
            logger.error(f"Error extracting usage info: {str(e)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def test_connection(self) -> bool:
        """
        Test connection to LLM API.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.is_configured:
            return False

        try:
            test_messages = [
                {"role": "user", "content": "Hello! This is a test message."}
            ]

            response = await self.generate_response(
                messages=test_messages,
                max_tokens=10,
            )

            return response is not None

        except Exception as e:
            logger.error(f"LLM connection test failed: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about configured model.

        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
            "is_configured": self.is_configured,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
        }