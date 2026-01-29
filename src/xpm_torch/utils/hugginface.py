from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, HfHubDownloadError
import logging

logger = logging.getLogger(__name__)

def download_huggingface_model(
    model_id: str,
    filename: str,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """
    Checks if a model file is present in the Hugging Face local cache.
    If not found, it downloads the model file from the Hugging Face Hub.

    Args:
        model_id (str): The model ID on the Hugging Face Hub (e.g., "bert-base-uncased").
        filename (str): The specific file name to download (e.g., "pytorch_model.bin", "config.json").
        subfolder (Optional[str]): A subfolder in the model repository where the file is located.
        revision (Optional[str]): The specific model version to use (branch name, tag name, or commit id).
        cache_dir (Optional[str]): Path to the folder where cached files are stored.

    Returns:
        Path: The local path to the downloaded (or already cached) model file.

    Raises:
        HfHubDownloadError: If the model file cannot be found locally or downloaded from the hub.
    """
    
    # First, try to load from local cache only
    try:
        local_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        logger.info(f"Model file '{filename}' for '{model_id}' found in local cache: {local_path}")
        return Path(local_path)
    except HfHubDownloadError:
        logger.info(f"Model file '{filename}' for '{model_id}' not found in local cache. Attempting download from Hugging Face Hub.")

    # If not in local cache, try to download from the hub
    try:
        local_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False, # Allow download
        )
        logger.info(f"Successfully downloaded model file '{filename}' for '{model_id}' to: {local_path}")
        return Path(local_path)
    except HfHubDownloadError as e:
        logger.error(f"Failed to download model file '{filename}' for '{model_id}' from Hugging Face Hub: {e}")
        raise

