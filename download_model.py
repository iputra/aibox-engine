#!/usr/bin/env python3
"""
Script to download Nomic embedding model.
"""

import os
import requests
from pathlib import Path

def download_file(url: str, destination: str) -> bool:
    """Download file from URL to destination."""
    try:
        print(f"Downloading from {url}...")
        print(f"Saving to {destination}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)

        print(f"\n✅ Downloaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False

def main():
    """Main function."""
    model_name = "nomic-embed-text-v1.5.Q4_K_M.gguf"
    model_url = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"
    model_path = f"models/{model_name}"

    if os.path.exists(model_path):
        print(f"✅ Model already exists at {model_path}")
        return

    print("📥 Downloading Nomic Embed Text v1.5 model...")
    print("This model is required for document embeddings and semantic search.")
    print("Model size: ~274MB")
    print()

    success = download_file(model_url, model_path)

    if success:
        print(f"\n✅ Model downloaded successfully!")
        print(f"Location: {model_path}")
        print("\nYou can now start the AIBox Engine and the embedding service will be available.")
    else:
        print("\n❌ Failed to download model.")
        print("Please download manually from:")
        print(model_url)
        print(f"And save to: {model_path}")

if __name__ == "__main__":
    main()