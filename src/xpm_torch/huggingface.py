"""HuggingFace Hub integration for xpm-torch.

Provides :class:`TorchHFHub` for exporting ModuleLoaders to the Hub
(calls ``write_hub_extras`` and ``hub_readme_sections``), plus utility
functions for cache checking and downloading.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union, Type, Dict, TypeVar, Any

from experimaestro.huggingface import ExperimaestroHFHub
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from experimaestro.core.context import SerializedPath
from experimaestro.core.objects import ConfigInformation
from experimaestro.core.objects.config import ConfigMixin

from xpm_torch.module import Module, ModuleLoader, assemble_readme_sections

import logging
logger = logging.getLogger(__name__)

# Generic variable that is either ModelHubMixin or a subclass thereof
T = TypeVar("T", bound="TorchHFHub")

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
        logger.info("All files for %s are already in cache.", model_id)
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
    """Check if all essential model or tokenizer files are downloaded in the cache.

    Queries HF API for the repository file manifest, verifies each essential
    file exists locally as a valid (non-broken) file, and touches each blob
    to reset Robinhood purge timers.

    Args:
        model_id: The ID of the model or tokenizer to check.
        is_model: Kept for signature compatibility.

    Returns:
        True if all essential files exist locally and are valid, False otherwise.
    """
    from huggingface_hub import HfApi

    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=model_id)

        # Ignore optional export formats & documentation metadata
        def is_essential(filename: str) -> bool:
            if filename.startswith(("onnx/", "openvino/", ".")):
                return False
            if filename.endswith((".onnx", ".gguf", ".ot", ".msgpack")):
                return False
            if filename in ("README.md", ".gitattributes", "results.csv", "experimaestro.json"):
                return False
            return True

        essential_files = [f for f in repo_files if is_essential(f)]
        if not essential_files:
            logger.warning("No essential files found for %s in HF Hub. Got only: %s", model_id, repo_files)
            return False

        for filename in essential_files:
            try:
                local_path = hf_hub_download(
                    repo_id=model_id, filename=filename, local_files_only=True
                )
                p = Path(local_path)
                # Verify that the file actually exists on disk (catches broken Robinhood symlinks)
                if not p.exists():
                    logger.warning(
                        "Broken symlink detected for %s in %s cache: %s",
                        filename,
                        model_id,
                        local_path,
                    )
                    return False

                # Touch the resolved target blob to reset Robinhood mtime/atime
                try:
                    p.resolve().touch()
                except Exception as te:
                    logger.debug("Failed to touch %s: %s", local_path, te)

            except (EntryNotFoundError, RepositoryNotFoundError, Exception):
                return False

        return True

    except Exception as e:
        logger.info("Unable to query HF API online for %s: %s", model_id, e)
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


class TorchHFHub(ExperimaestroHFHub):
    """HF Hub integration for xpm-torch ModuleLoaders.

    Extends :class:`~experimaestro.huggingface.ExperimaestroHFHub` to call
    :meth:`~xpm_torch.module.ModuleLoader.write_hub_extras` after
    serialization and build the README.
    :meth:`~xpm_torch.module.ModuleLoader.hub_readme_sections`.

    Subclass this (e.g. ``XPMIRHFHub``) to add library-specific README
    sections, TensorBoard logs, etc.
    """

    def __init__(self, loader: ModuleLoader):
        #Just rename the "config" attribute to "loader" for clarity, 
        # since this class is specifically for ModuleLoaders
        super().__init__(loader.__config__)
        self.loader = loader

    def _save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        super()._save_pretrained(save_directory)

        # Call ModuleLoader hub hooks
        self.loader.write_hub_extras(save_directory)

        # Build README from loader sections only if README.md was not generated by write_hub_extras
        if not (save_directory / "README.md").exists():
            loader_sections = self.loader.hub_readme_sections()
            base_sections = self._readme_base_sections()
            if base_sections or loader_sections:
                readme = assemble_readme_sections(base_sections, loader_sections)
                (save_directory / "README.md").write_text(readme)
        else:
            logger.info(
                "README.md already exists in %s, skipping README generation.",
                save_directory,
            )

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=None,
        local_files_only=False,
        token=None,
        **model_kwargs
    ) -> Module:
        """
        This overrides `ExperimaestroHFHub._from_pretrained`
        outputs directly the model instance instead of the loader.
        """
        if os.path.isdir(model_id):
            save_directory = Path(model_id)

            def loader_path(path: Path):
                if isinstance(path, SerializedPath):
                    path = path.path
                else:
                    path = Path(path)
                return save_directory / path

        else:

            def loader_path(s_path: Union[Path, str, SerializedPath]) -> Path:
                if not isinstance(s_path, SerializedPath):
                    s_path = SerializedPath(Path(s_path), False)
                path = s_path.path

                # Folder
                if s_path.is_folder:
                    hf_path = snapshot_download(
                        repo_id=model_id,
                        allow_patterns=f"{s_path.path}/**",
                        revision=revision,
                        cache_dir=cache_dir,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    return Path(hf_path) / path

                hf_path = Path(
                    hf_hub_download(
                        repo_id=model_id,
                        filename=str(path),
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                )
                return hf_path

        loader: ModuleLoader = ConfigInformation.deserialize(
            loader_path,
            as_instance=True,
            partial_loading=True,
            definition_filename=cls.definition_filename,
        )
        #execute the moduleLoader Instance -> loads the model
        loader.execute()

        return loader.model

    @classmethod
    def pretrained_loader(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        *,
        force_download: bool = False,
        resume_download: Optional[bool] = None,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        as_instance: bool = False, #specific to this Class
        **model_kwargs,
    ) -> ModuleLoader:
        """
        Download a model _loader_ from the Huggingface Hub.
        """
        # Call parent's _from_pretrained directly to avoid the overridden version
        return ExperimaestroHFHub.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            as_instance=as_instance, # pass to super but it will be ignored
            **model_kwargs,
        )


    def _readme_base_sections(self):
        """Return base README sections. Override in subclasses to add
        library-specific content (description, usage examples, results)."""
        return []
