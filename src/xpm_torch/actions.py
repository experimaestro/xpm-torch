"""Experiment actions for xpm-torch.

Provides :class:`ExportAction` for exporting trained models to
HuggingFace Hub or local directories after experiment completion.
"""

from pathlib import Path
from typing import Optional, Union

from experimaestro import Action, Interaction, Param, field
from experimaestro.annotations import tags as get_tags

from xpm_torch.module import ModuleLoader


class DirectInteraction(Interaction):
    """Non-interactive interaction backend for pre-filled programmatic action execution."""

    def __init__(
        self,
        target: str = "Local folder",
        folder: Optional[Union[str, Path]] = None,
        repo_id: Optional[str] = None,
        private: bool = False,
    ):
        self._target = target
        self._folder = str(folder) if folder else ""
        self._repo_id = repo_id or ""
        self._private = private

    def choice(self, key: str, label: str, choices: list) -> str:
        return self._target

    def text(self, key: str, label: str, *, default: str = "") -> str:
        if key == "folder":
            return self._folder
        elif key == "repo_id":
            return self._repo_id or default
        return default

    def checkbox(self, key: str, label: str, *, default: bool = False) -> bool:
        if key == "private":
            return self._private
        return default


# TODO: shouldn't this be a Task ? 
class ExportAction(Action):
    """Export a trained model to HuggingFace Hub or a local directory.

    Uses :meth:`get_hub` to obtain the HF Hub wrapper. Subclass and
    override :meth:`get_hub` to use a library-specific hub class
    (e.g. ``XPMIRHFHub`` for xpmir models).
    """

    loader: Param[ModuleLoader]
    """The model loader to export"""

    default_name: Param[str] = field(default="", ignore_default=True)
    """Default HF Hub model name (for pre-fill)"""

    def get_hub(self):
        """Return the HF Hub wrapper for this loader.

        Override in subclasses to use a library-specific hub class.
        """
        from xpm_torch.huggingface import TorchHFHub

        return TorchHFHub(self.loader)

    def _tags_str(self) -> str:
        """Build a description string from the loader's tags."""
        try:
            tags = get_tags(self.loader)
            if tags:
                return ", ".join(f"{k}={v}" for k, v in tags.items())
        except Exception:
            pass
        return ""

    def describe(self) -> str:
        parts = ["Export"]
        tags = self._tags_str()
        if self.default_name:
            parts.append(f"'{self.default_name}'")
        if tags:
            parts.append(f"({tags})")
        parts.append("to HuggingFace Hub or local directory")
        return " ".join(parts)

    def execute(self, interaction: Interaction) -> None:
        self.loader.execute()

        hub = self.get_hub()

        target = interaction.choice(
            "target", "Export to:", ["HF Hub", "Local folder"]
        )

        if target == "HF Hub":
            repo_id = interaction.text(
                "repo_id",
                "HF Hub repo ID (e.g. user/model-name):",
                default=self.default_name,
            )
            private = interaction.checkbox("private", "Private repo?", default=False)
            hub.push_to_hub(repo_id=repo_id, private=private)
        else:
            folder = interaction.text("folder", "Output folder:")
            path = Path(folder)
            path.mkdir(parents=True, exist_ok=True)
            if self.default_name:
                self.loader.model_name = self.default_name
                self.loader.value.model_name = self.default_name
            hub.save_pretrained(path)

    def export_to_folder(self, folder: Union[str, Path]) -> None:
        """Helper to execute model export non-interactively to a local folder."""
        interaction = DirectInteraction(target="Local folder", folder=folder)
        self.execute(interaction)

    def push_to_hub_direct(self, repo_id: str, private: bool = False) -> None:
        """Helper to execute model export non-interactively to HF Hub."""
        interaction = DirectInteraction(
            target="HF Hub", repo_id=repo_id, private=private
        )
        self.execute(interaction)
