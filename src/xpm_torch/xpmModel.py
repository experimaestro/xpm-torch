from typing import (
    Dict,
    Type,
    TypeVar,
    Union,
    Optional,
)
import torch, os, json, logging
import torch.nn as nn
from experimaestro.core.arguments import Meta
from experimaestro import (
    Config,
    Task,
    LightweightTask,
    Param,
    DataPath,
    serialize,
    deserialize,
)
from experimaestro.core.context import SerializedPath
from pathlib import Path
from huggingface_hub import ModelHubMixin, snapshot_download, hf_hub_download


T = TypeVar("T", bound="xpmTorchHubModule")


class xpmTorchHubModule(nn.Module, Config, ModelHubMixin):
    """
    Generic PyTorch module for experimaestro, compatible with Hugging Face Hub.

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

    _config: dict
    """Module *initial* configuration, can be used to recreate module"""

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: dict):
        if self._initialized:
            raise AttributeError(
                "config is immutable and cannot be modified after initialization"
            )
        self._config = config

    def get_xpm_cfg(self):
        return self.C(**self.config)

    class Loader(LightweightTask):
        """Loader for the xpmTorchHubModule"""

        model: Param["xpmTorchHubModule"]
        parameters: Meta[DataPath]

        def execute(self):
            """Loads the model from disk using the given serialization path"""
            state_dict = torch.load(self.parameters)
            self.model.load_state_dict(state_dict)

    def __post_init__(self):
        super().__post_init__()
        self._initialized = False
        self.config = {name: getattr(self, name) for name in self.__xpmtype__.arguments}
        self._initialized = True

    @property
    def device(self):
        return next(self.parameters()).device

    def _save_pretrained(self, save_directory: Path):
        """
        Save the model weights and config to the specified directory.
        """

        # Save model weights to directory
        params_path = Path("parameters.pth")
        torch.save(self.state_dict(), params_path)
        xpm_cfg = self.get_xpm_cfg()

        # serialize whole model and dependancies
        serialize(
            xpm_cfg,
            save_directory,
            init_tasks=[self.Loader.C(model=xpm_cfg, parameters=params_path)],
        )

        # Save initial config (will likely have issues serializing parent models)
        with open(save_directory / "config.json", "w") as f:
            json.dump(self.config, f)

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
            save_directory = Path(model_id)

            return deserialize(save_directory, as_instance=True)

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

    @classmethod
    def from_kwargs(cls, **kwargs) -> "xpmTorchHubModule":
        """
        Alternate constructor that parses kwargs into xpmConfig and returns an instance.
        """
        cfg = cls.C(**kwargs)
        return cfg.instance()


# Example child module
class MyTorchxpmTorchHubModule(
    xpmTorchHubModule,
    library_name="my-org/my-model",
    tags=["example", "torch"],
    repo_url="https://huggingface.co/my-org/my-model",
    paper_url="https://arxiv.org/abs/???",
):
    """Example xpmTorchHubModule implementation, now all saving and loading is taken care of under the hood"""

    ## Child parameters
    input_dim: Param[int] = 100
    """Input dimension"""

    hidden_dim: Param[int] = 200
    """Hidden dimension"""

    output_dim: Param[int] = 10
    """Output dimension"""

    def __post_init__(self):
        super().__post_init__()
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        logging.debug("Initialized layers: %s, %s", self.fc1, self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    model_id = "test-model"
    save_path = Path("./model")

    create_new_model = True
    create_new_model = False

    if create_new_model:
        # create configuration
        cfg = MyTorchxpmTorchHubModule.C(input_dim=50, hidden_dim=100, output_dim=5)
        # model = xpmTorchHubModule.from_kwargs(input_dim=50, hidden_dim=100, output_dim=5)
        # model = xpmTorchHubModule(input_dim=50, hidden_dim=100, output_dim=5)

        model: MyTorchxpmTorchHubModule = cfg.instance()
        print(model)

        # test it
        input = torch.randn(1, 50)
        output = model(input)

        print(output)

        model.save_pretrained(save_path)
        model.push_to_hub(model_id)

    # save_path = Path("./test_model")
    # try to load from Hub

    newModel = MyTorchxpmTorchHubModule.from_pretrained(
        model_id, force_download=True, local_files_only=False
    )

    logging.info("New model: %s", newModel)
    logging.info("New model config: %s", newModel.config)