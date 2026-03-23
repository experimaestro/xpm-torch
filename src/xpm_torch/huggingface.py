"""HuggingFace Hub utility functions for xpm-torch.

Provides helpers for checking cache, downloading models, and reading configs
from HuggingFace Hub without importing the full transformers library.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


@lru_cache
def prepare_hf_model(model_id: str) -> bool:
    """Check if model and tokenizer are in cache, if not, download all necessary files.

    Args:
        model_id: The ID of the model to check.

    Returns:
        True if both model and tokenizer are in cache or after downloading,
        False if download fails.
    """
    model_in_cache = check_hf_cache(model_id, is_model=True)
    tokenizer_in_cache = check_hf_cache(model_id, is_model=False)

    logger.info("Preparing model %s ...", model_id)
    if model_in_cache and tokenizer_in_cache:
        logger.info("Model and tokenizer for %s are already in cache.", model_id)
        return True

    logger.info("Downloading missing files for %s...", model_id)
    try:
        if not model_in_cache:
            snapshot_download(repo_id=model_id)
        if not tokenizer_in_cache:
            snapshot_download(repo_id=model_id)
        logger.info("Successfully downloaded missing files for %s.", model_id)
        return True
    except Exception as e:
        logger.error("Failed to download files for %s: %s", model_id, e)
        return False


def check_hf_cache(model_id: str, is_model: bool = True) -> bool:
    """Check if the model or tokenizer is already downloaded in the cache.

    Args:
        model_id: The ID of the model or tokenizer to check.
        is_model: If True, checks for model files. If False, checks for tokenizer files.

    Returns:
        True if the model or tokenizer is already downloaded, False otherwise.
    """
    model_files = [
        "config.json",
        "pytorch_model.bin",
        "tf_model.h5",
        "model.safetensors",
    ]
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    files_to_check = model_files if is_model else tokenizer_files

    for filename in files_to_check:
        try:
            hf_hub_download(
                repo_id=model_id, filename=filename, local_files_only=True
            )
            return True
        except (EntryNotFoundError, RepositoryNotFoundError):
            continue

    return False


def get_hf_config(repo_id: str) -> dict:
    """Pull config.json from HF Hub without importing transformers."""
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path) as f:
        return json.load(f)


def download_huggingface_model(
    model_id: str,
    filename: str,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Download a model file from HuggingFace Hub, using local cache if available.

    Args:
        model_id: The model ID on HuggingFace Hub (e.g., "bert-base-uncased").
        filename: The specific file name to download.
        subfolder: A subfolder in the model repository where the file is located.
        revision: The specific model version to use.
        cache_dir: Path to the folder where cached files are stored.

    Returns:
        The local path to the downloaded (or already cached) model file.

    Raises:
        ValueError: If the model file cannot be found locally or downloaded.
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
        logger.info(
            "Model file '%s' for '%s' found in local cache: %s",
            filename,
            model_id,
            local_path,
        )
        return Path(local_path)
    except ValueError:
        logger.info(
            "Model file '%s' for '%s' not found in local cache. "
            "Attempting download from HuggingFace Hub.",
            filename,
            model_id,
        )

    # If not in local cache, try to download from the hub
    try:
        local_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        logger.info(
            "Successfully downloaded model file '%s' for '%s' to: %s",
            filename,
            model_id,
            local_path,
        )
        return Path(local_path)
    except ValueError as e:
        logger.error(
            "Failed to download model file '%s' for '%s' from HuggingFace Hub: %s",
            filename,
            model_id,
            e,
        )
        raise
