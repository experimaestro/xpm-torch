from dataclasses import dataclass
from typing import (
    List,
    Dict,
    Optional,
)
from pathlib import Path
import torch
import logging
import torch.nn as nn
from experimaestro import (
    Config,
    PathSerializationLWTask,
    Param,
)
from xpm_torch.utils.utils import Initializable

logger = logging.getLogger(__name__)


def initialized(method):
    """Decorator that ensures ``initialize()`` is called before the first
    invocation, then replaces itself with the original method so subsequent
    calls have zero overhead.

    Usage::

        class MyModule(Module):
            @initialized
            def forward(self, x):
                ...
    """

    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            self.initialize()
        # Replace the wrapper with the unwrapped method on this instance
        bound = method.__get__(self, type(self))
        setattr(self, method.__name__, bound)
        return method(self, *args, **kwargs)

    # Preserve the original name/docstring for introspection
    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


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

    def save_model(self, path: Path):
        """Save model parameters to a directory using safetensors."""
        from safetensors.torch import save_file

        path.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(path / "model.safetensors"))

    def load_model(self, path: Path):
        """Load model parameters from a directory."""
        from safetensors.torch import load_file

        self.load_state_dict(load_file(str(path / "model.safetensors")))

    def loader_config(self, path: Path) -> "ModuleLoader":
        """Returns a ModuleLoader config that knows how to load this model from path.

        The loader handles DataPath fields internally. For a simple Module,
        this is a single DataPath to the model/ directory. Subclasses like
        DotDense override to return custom loaders with multiple DataPaths
        (one per sub-encoder).

        Args:
            path: The base checkpoint path containing the model/ directory.
        """
        return ModuleLoader.C(value=self, path=path)

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


@dataclass
class ReadmeSection:
    """A named section for the HF Hub README.

    Sections are assembled in order, with optional ``before``/``after``
    constraints for positioning relative to other sections.
    """

    key: str
    """Unique identifier for this section."""

    content: str
    """Markdown content of this section."""

    before: Optional[str] = None
    """Insert this section before the section with this key."""

    after: Optional[str] = None
    """Insert this section after the section with this key."""


def assemble_readme_sections(
    base: List[ReadmeSection], extra: List[ReadmeSection]
) -> str:
    """Merge extra sections into base using before/after constraints,
    then concatenate all contents."""
    sections = list(base)
    for s in extra:
        if s.before:
            idx = next(
                (i for i, b in enumerate(sections) if b.key == s.before),
                len(sections),
            )
            sections.insert(idx, s)
        elif s.after:
            idx = next(
                (i for i, b in enumerate(sections) if b.key == s.after),
                len(sections) - 1,
            )
            sections.insert(idx + 1, s)
        else:
            sections.append(s)
    return "\n".join(s.content for s in sections)


class ModuleLoader(PathSerializationLWTask):
    """Loads a model from a checkpoint directory.

    Subclasses override :meth:`write_hub_extras` and
    :meth:`hub_readme_sections` to customize what gets written when the
    model is exported to HuggingFace Hub.

    The model config is accessible via :attr:`model` (alias for ``value``).
    """

    @property
    def model(self):
        """The model config (alias for ``value``)."""
        return self.value

    def write_hub_extras(self, save_directory: Path):
        """Write additional files when exporting to HuggingFace Hub.

        Called by ``ExperimaestroHFHub._save_pretrained`` after the main
        serialization. Override in subclasses to write format-specific
        files (e.g. sentence-transformers configs).

        Args:
            save_directory: The Hub export directory.
        """
        pass

    def hub_readme_sections(self) -> List[ReadmeSection]:
        """Return additional sections for the HF Hub README.

        Each :class:`ReadmeSection` has a key and content, plus optional
        ``before``/``after`` constraints for positioning relative to the
        base sections (``frontmatter``, ``description``, ``usage``,
        ``results``).

        Override in subclasses to provide model-specific content.
        """
        return []

    def execute(self):
        """Loads the model from disk using the given serialization path"""
        # First initialize model structure (empty init)
        self.value.initialize()
        # Then load weights: try model/ directory first, fall back to model.pth
        path = Path(self.path)
        logger.info("Loading model from disk: %s", path)
        model_dir = path / "model"
        if model_dir.exists():
            self.value.load_model(model_dir)
        else:
            data = torch.load(path / "model.pth", map_location="cpu", weights_only=True)
            self.value.load_state_dict(data)


class ModuleContainer(nn.Module):
    """A container for Modules, exposing only nn.Module attributes
    that actually contain state (parameters or buffers).

    Example::

        class MyRetriever(ModuleContainer):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(128, 64)  # Has params -> wrapped
                self.activ = nn.ReLU()              # No params -> ignored

        retriever = MyRetriever()
        retriever.setup_with_fabric(fabric)
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

    def setup_with_fabric(self, fabric) -> None:
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
            logger.debug(f"Registered {name} (type: {type(module).__name__}) with Fabric on {fabric.device}")


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

