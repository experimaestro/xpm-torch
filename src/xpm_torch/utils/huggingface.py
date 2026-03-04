from functools import lru_cache
from pathlib import Path
import json
from typing import Optional
from huggingface_hub import PyTorchModelHubMixin, snapshot_download, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
import torch, os, logging, shutil
from typing import (
    List,
    Dict,
    Type,
    TypeVar,
    Union,
    Optional,
)

from experimaestro import (
    LightweightTask,
    Param,
    Meta,
    Constant,
    DataPath,
    serialize,
    deserialize,
)
from experimaestro.core.context import SerializedPath
from xpm_torch.module import Module





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




T = TypeVar("T", bound="xpmTorchHubModule")


class xpmTorchHubModule(Module, PyTorchModelHubMixin):
    """
    Generic PyTorch module for experimaestro, 
    compatible with Hugging Face Hub via [ModelHubMixin](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins).

    Example usage:
    ```python
    class MyTorchxpmTorchHubModule(xpmTorchHubModule,
        library_name="my-org/my-model",
        tags=["example", "torch"],
        repo_url="https://huggingface.co/my-org/my-model",
        paper_url="https://arxiv.org/abs/???"
        ):

        ## Child parameters
        input_dim: Param[int] = 100

        hidden_dim: Param[int] = 200

        output_dim: Param[int] = 10

        def __post_init__(self):
            # Here we use the child parameters to initialize our module layers and other stuff
            super().__post_init__()
            self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
            logging.debug("Initialized layers: %s, %s", self.fc1, self.fc2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    #new model configuration
    model_cfg:Config = MyTorchxpmTorchHubModule.C(input_dim=50, hidden_dim=100, output_dim=5)

    #instanciate the model fomr config
    model = model_cfg.instance()

    # Saving
    #save to local directory
    model.save_pretrained("my_model_dir")
    #push to HF hub
    model.push_to_hub("my_model_id")

    # Loading
    #load from file
    new_model = MyTorchxpmTorchHubModule.from_pretrained("my_model_dir")
    #load from HF hub
    new_model = MyTorchxpmTorchHubModule.from_pretrained("my_model", force_download=True)
    ```
    """

    class Loader(LightweightTask):
        """Loader for the xpmTorchHubModule"""

        model: Param["xpmTorchHubModule"]
        parameters: Meta[DataPath]

        def execute(self):
            """Loads the model from disk using the given serialization path"""
            state_dict = torch.load(self.parameters, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def __post_init__(self):
        """Initialize the module. Child classes should override this method to initialize their layers and other stuff."""
        super().__post_init__()
    
    @classmethod
    def from_kwargs(cls, **kwargs) -> "xpmTorchHubModule":
        """
        Alternate constructor that parses kwargs into xpmConfig and returns an instance.
        """
        cfg = cls.C(**kwargs)
        return cfg.instance()

    def extra_repr(self):
        res = super().extra_repr()
        if hasattr(self, "_parameters"):
            res += f", n_params={self.count_parameters()}"
        return res

    def customize_hf_serialization(self, hf_serialization):
        """Saves the model and tokenizer in a way that they can be loaded back
        using the HuggingFace Transformers library. This allows to use the model
        in HuggingFace pipelines, and to share it easily with the community."""

        # TODO: when XPM torch stabilizes
        raise NotImplementedError
    
    @property
    def device(self):
        return next(self.parameters()).device

    def count_parameters(self):
        """Count the number of parameters in the model"""
        if not hasattr(self, "_parameters"):
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # def save_pretrained(
    #     self,
    #     save_directory,
    #     *,
    #     config=None,
    #     repo_id=None,
    #     push_to_hub=False,
    #     model_card_kwargs=None,
    #     **push_to_hub_kwargs,
    # ):
    #     """ Save the model weights and config to the specified directory.
    #     see <https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.ModelHubMixin>
    #     """

    #     return super().save_pretrained(
    #         save_directory,
    #         config=config,
    #         repo_id=repo_id,
    #         push_to_hub=push_to_hub,
    #         model_card_kwargs=model_card_kwargs,
    #         **push_to_hub_kwargs,
    #     )

    def _save_pretrained(self, save_directory: Path):
        """
        Save the model weights and config to the specified directory.
        """

        # Save model weights to directory
        params_path = Path("parameters.pth")
        torch.save(self.state_dict(), params_path)
        if save_directory.exists():
            logging.warning(f"Folder {save_directory} already exists, overwriting it")
            shutil.rmtree(save_directory)
        # serialize whole model and dependancies
        xpm_config = self.__config__
        serialize(
            xpm_config,
            save_directory,
            init_tasks=[self.Loader.C(model=xpm_config, parameters=params_path)],
        )

        # # finally delete temporary files
        # params_path.unlink(missing_ok=True)
    
    @classmethod
    def _from_pretrained(
        cls: Type[T],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> T:
        """
        Load the model weights and config from the specified directory.
        Rest of loading logic is done by HF ModelHubMixin.
        see <https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.ModelHubMixin>
        """
        if os.path.isdir(model_id):
            #model is save Locally
            save_directory = Path(model_id)
            model_cfg, [loader] = deserialize(save_directory, as_instance=False)
            #get Loader Task
            loader = loader.instance()
            loader.execute()
            model = loader.model
            object.__setattr__(model, "__config__", model_cfg)
            return model

        else:

            def data_loader(s_path: Union[Path, str, SerializedPath]):
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

        return deserialize(data_loader, as_instance=True)

