"""Experiment actions for xpm-torch.

Provides :class:`ExportAction` for exporting trained models to
HuggingFace Hub or local directories after experiment completion.
"""

from pathlib import Path

from experimaestro import Action, Interaction, Param, field
from experimaestro.annotations import tags as get_tags

from xpm_torch.module import ModuleLoader


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
            hub.save_pretrained(path)
