from typing import (
    List,
    Dict,
    Type,
    TypeVar,
    Union,
    Optional,
)
from pathlib import Path
import torch, logging
import torch.nn as nn
from experimaestro import (
    Config,
    PathSerializationLWTask,
    Task,
    LightweightTask,
    Param,
    Meta,
    Constant,
    DataPath,
    serialize,
    deserialize,
)
from xpm_torch.utils.utils import Initializable

logger = logging.getLogger(__name__)

class Module(Config, Initializable, nn.Module):
    """Base class for all modules containing parameters"""

    def __init__(self):
        Initializable.__init__(self)
        torch.nn.Module.__init__(self)


    def __initialize__(self):
        """Initialize a module (structure only, no weight loading)"""
        pass

    def __call__(self, *args, **kwargs):
        return torch.nn.Module.__call__(self, *args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        return torch.nn.Module.to(self, *args, **kwargs)


class ModuleList(Module, Initializable):
    """Groups different models together, to be used within the Learner"""

    sub_modules: Param[List[Module]]

    def __post_init__(self):
        # Register sub-modules
        for ix, sub_module in enumerate(self.sub_modules):
            self.add_module(str(ix), sub_module)

    def __initialize__(self):
        for module in self.sub_modules:
            module.initialize()

    def __call__(self, *args, **kwargs):
        raise AssertionError("This module cannot be used as such")

    def to(self, *args, **kwargs):
        return torch.nn.Module.to(self, *args, **kwargs)


class ModuleLoader(PathSerializationLWTask):
    def execute(self):
        """Loads the model from disk using the given serialization path"""
        logger.info("Loading model from disk: %s", self.path)
        self.value.initialize()
        data = torch.load(self.path)

        # Check if model has the dummy param; if not, remove it from data
        if "_dummy_param" in data and "_dummy_param" not in self.value.state_dict():
            logger.debug(
                "Ignoring '_dummy_param' as it is not present in the model architecture."
            )
            data.pop("_dummy_param")

        self.value.load_state_dict(data)


class ModuleContainer(nn.Module):
    """
    A config that can contain Modules, 
    exposing only nn.Module attributes if they actually contain state (parameters or buffers).
    ```py
    # example implementation
    class MyRetriever(ModuleContainer):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(128, 64) # Has params -> Will be wrapped
            self.activ = nn.ReLU()            # No params -> Will be ignored by fabric.setup

    # Usage
    retriever = MyRetriever()
    retriever.setup_with_fabric(fabric)
    ```
    """

    def __init__(self):
        super().__init__()

    def get_manageable_modules(self) -> Dict[str, nn.Module]:
        """
        Returns a mapping of attributes that are nn.Modules
        AND have actual data (params/buffers) to manage.
        """
        manageable = {}
        # Iterate through immediate children, thanks to nn.Module registering
        for name, module in self.named_children():
            # Check if this specific module or any of its descendants have state
            has_params = any(p.numel() > 0 for p in module.parameters())
            has_buffers = any(b.numel() > 0 for b in module.buffers())

            if has_params or has_buffers:
                manageable[name] = module

        return manageable

    def setup_with_fabric(self, fabric):
        """
        Self-identifies which children need Fabric wrapping.
        """
        modules_to_wrap = self.get_manageable_modules()

        if not modules_to_wrap:
            logger.debug("No stateful modules found. Skipping Fabric setup.")
            return

        for name, module in modules_to_wrap.items():
            # Wrap the module and re-assign it
            wrapped = fabric.setup(module)
            setattr(self, name, wrapped)
            logger.debug(f"Registered {name} with Fabric on {fabric.device}")


def find_module_attributes(obj) -> dict:
    """
    Finds all instances of `xpm_torch.Module` in attributes of any object.
    only looks at immediate attributes, not recursively.
    """
    found_modules = {}

    # vars(obj) returns the __dict__ of the instance
    # We use list() to avoid "dictionary changed size" errors if needed
    for attr_name, attr_value in vars(obj).items():
        if isinstance(attr_value, Module):
            found_modules[attr_name] = attr_value

    return found_modules

