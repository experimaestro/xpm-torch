from functools import lru_cache
from pathlib import Path
import json
from typing import Optional
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
import logging

logger = logging.getLogger(__name__)

@lru_cache
def prepare_hf_model(model_id: str) -> bool:
    """Check if model and tokenizer are in cache, if not, download all necessary files.

    Args:
        model_id: The ID of the model to check.

    Returns:
        True if both model and tokenizer are in cache or after downloading, False if download fails.
    """
    model_in_cache = check_hf_cache(model_id, is_model=True)
    tokenizer_in_cache = check_hf_cache(model_id, is_model=False)

    logger.info(f"Preparing model {model_id} ...")
    if model_in_cache and tokenizer_in_cache:
        logger.info(f"Model and tokenizer for {model_id} are already in cache.")
        return True
    else:
        logger.info(f"Downloading missing files for {model_id}...")
        try:
            # Download model files if not in cache
            if not model_in_cache:
                snapshot_download(repo_id=model_id)

            # Download tokenizer files if not in cache
            if not tokenizer_in_cache:
                snapshot_download(repo_id=model_id)
                
            logger.info(f"Successfully downloaded missing files for {model_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to download files for {model_id}: {e}")
            return False

def check_hf_cache(model_id: str, is_model: bool = True) -> bool:
    """Check if the model or tokenizer is already downloaded in the cache.

    Args:
        model_id: The ID of the model or tokenizer to check.
        is_model: If True, checks for model files. If False, checks for tokenizer files.

    Returns:
        True if the model or tokenizer is already downloaded, False otherwise.
    """
    model_files = ["config.json", "pytorch_model.bin", "tf_model.h5", "model.safetensors"]
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]

    files_to_check = model_files if is_model else tokenizer_files

    for filename in files_to_check:
        try:
            hf_hub_download(repo_id=model_id, filename=filename, local_files_only=True)
            return True
        except (EntryNotFoundError, RepositoryNotFoundError):
            continue

    return False


def get_hf_config(repo_id: str) -> dict:
    """ pull config from HF, don't need to import transformers"""
    # Download the config.json file to your local cache
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    
    # Load and parse the JSON
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        
    return config_dict

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
        ValueError: If the model file cannot be found locally or downloaded from the hub.
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
    except ValueError:
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
    except ValueError as e:
        logger.error(f"Failed to download model file '{filename}' for '{model_id}' from Hugging Face Hub: {e}")
        raise

