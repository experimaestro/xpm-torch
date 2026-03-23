HuggingFace Hub Integration
===========================

xpm-torch models can be pushed to and loaded from the
`HuggingFace Hub <https://huggingface.co/>`_ via experimaestro's
``ExperimaestroHFHub`` class. The serialization system preserves the full
experimaestro configuration graph, so a downloaded model can be used
directly in further experiments.

Pushing a model to the Hub
--------------------------

After training, use ``ExperimaestroHFHub`` to upload a model config
(typically a :class:`~xpm_torch.module.ModuleLoader` or the model config
itself):

.. code-block:: python

    from experimaestro.huggingface import ExperimaestroHFHub

    # Push a model config (e.g. a trained scorer)
    ExperimaestroHFHub(model_config).push_to_hub("your-org/model-name")

    # With a private repo
    ExperimaestroHFHub(model_config).push_to_hub(
        "your-org/model-name",
        private=True,
    )

What gets uploaded
~~~~~~~~~~~~~~~~~~

``ExperimaestroHFHub`` serializes the full experimaestro config tree.
For a :class:`~xpm_torch.module.ModuleLoader` (produced by
:meth:`~xpm_torch.module.Module.loader_config`), this includes:

- ``experimaestro.json`` — the config definition
- ``model/`` — directory containing model weights (safetensors via
  :meth:`~xpm_torch.module.Module.save_model`)

  - For dual-encoder models, this contains ``encoder/`` and optionally
    ``query_encoder/`` subdirectories

Saving locally first
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Save to a local directory (same format as Hub upload)
    ExperimaestroHFHub(model_config).save_pretrained("/path/to/save")

Loading a model from the Hub
-----------------------------

Low-level (experimaestro):

.. code-block:: python

    from experimaestro.huggingface import ExperimaestroHFHub

    # Returns a deserialized config (e.g. ModuleLoader)
    data = ExperimaestroHFHub.from_pretrained("your-org/model-name")

For xpmir models, use ``AutoModel`` which returns a
:class:`~xpm_torch.module.ModuleLoader` directly:

.. code-block:: python

    from xpmir.models import AutoModel

    # Returns a ModuleLoader — use as an init task in experiments
    loader = AutoModel.load_from_hf_hub("your-org/model-name")
    # loader.model is the model config

    # Or load as a ready-to-use instance for direct inference
    model = AutoModel.load_from_hf_hub("your-org/model-name", as_instance=True)

Using the CLI
-------------

xpm-torch provides a CLI command to export trained models from an
experimaestro workspace:

.. code-block:: bash

    # Interactive mode — prompts for workspace, experiment, and model
    xpm-torch upload-hfhub

    # Non-interactive
    xpm-torch upload-hfhub \
        --workdir /path/to/workspace \
        --experiment my-experiment \
        --model-key my-model \
        --repo-id your-org/model-name

    # Save locally instead of uploading
    xpm-torch upload-hfhub --save-dir /path/to/output

The CLI looks for a :class:`~xpm_torch.results.TrainingResults` that was
persisted by the experiment at the end of its run. Experiments save it
with ``xp.save(training_results, "xpm-torch-models")``, which writes a
serialized config under
``<workspace>/experiments/<experiment>/current/data/xpm-torch-models/``.

The CLI deserializes this config, lets you pick a model key from
:attr:`~xpm_torch.results.TrainingResults.models`, and then calls
``ExperimaestroHFHub(model).push_to_hub(...)`` (or
``save_pretrained(...)`` for local export).

Customizing the HF checkpoint format
-------------------------------------

There are three extension points for customizing what gets written during
Hub export. They live on different classes:

- :meth:`Module.loader_config(path) <xpm_torch.module.Module.loader_config>` —
  on the model, controls which ``ModuleLoader`` subclass is returned
- :meth:`ModuleLoader.write_hub_extras(save_directory) <xpm_torch.module.ModuleLoader.write_hub_extras>` —
  on the loader, writes additional files (e.g. ST configs)
- :meth:`ModuleLoader.hub_readme_extra() <xpm_torch.module.ModuleLoader.hub_readme_extra>` —
  on the loader, appends model-specific text to the README

The hooks are on :class:`~xpm_torch.module.ModuleLoader` (not on
:class:`~xpm_torch.module.Module`) because the loader is the object that
gets serialized for Hub export and holds the path to the model weights.
``Module`` configs are data-less.

:meth:`~xpm_torch.module.Module.loader_config`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a model is serialized for the Hub, experimaestro serializes the
:class:`~xpm_torch.module.ModuleLoader` returned by
:meth:`~xpm_torch.module.Module.loader_config`. Subclasses override this
method to return a custom loader:

.. code-block:: python

    from xpm_torch.module import Module, ModuleLoader

    class MyModel(Module):
        def loader_config(self, path):
            # Default: single ModuleLoader pointing to checkpoint dir
            return ModuleLoader.C(value=self, path=path)

    class MySpladeModel(Module):
        def loader_config(self, path):
            # Return a custom loader that writes ST configs on Hub export
            return SpladeModuleLoader.C(value=self, path=path)

:meth:`~xpm_torch.module.Module.save_model` / :meth:`~xpm_torch.module.Module.load_model`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods on :class:`~xpm_torch.module.Module` control how model
weights are written to and read from a directory. Override them to change
the on-disk format:

.. code-block:: python

    class DualEncoderModel(Module):
        encoder: Param[Module]
        query_encoder: Param[Optional[Module]]

        def save_model(self, path):
            path.mkdir(parents=True, exist_ok=True)
            self.encoder.save_model(path / "encoder")
            if self.query_encoder is not None:
                self.query_encoder.save_model(path / "query_encoder")

        def load_model(self, path):
            self.encoder.load_model(path / "encoder")
            if (path / "query_encoder").exists():
                self.query_encoder.load_model(path / "query_encoder")

:meth:`~xpm_torch.module.ModuleLoader.write_hub_extras` and :meth:`~xpm_torch.module.ModuleLoader.hub_readme_sections`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To write additional files alongside the model weights during Hub export
(e.g. sentence-transformers compatibility configs), create a custom
:class:`~xpm_torch.module.ModuleLoader` subclass and override
:meth:`~xpm_torch.module.ModuleLoader.write_hub_extras`.

To add sections to the README, override
:meth:`~xpm_torch.module.ModuleLoader.hub_readme_sections` and return a
list of :class:`~xpm_torch.module.ReadmeSection`. Each section has a
``key``, ``content``, and optional ``before``/``after`` constraints for
positioning relative to the base sections (``frontmatter``,
``description``, ``usage``, ``results``):

.. code-block:: python

    from xpm_torch.module import ModuleLoader, ReadmeSection

    class MyCustomLoader(ModuleLoader):
        def write_hub_extras(self, save_directory):
            (save_directory / "my_config.json").write_text('{"format": "custom"}')

        def hub_readme_sections(self):
            return [
                ReadmeSection(
                    key="quick_loading",
                    content="## Quick loading\n\n```python\nmodel = MyLib.load(...)\n```",
                    before="usage",  # appears before the XPMIR usage section
                ),
            ]

These hooks are only called during Hub export (by ``ExperimaestroHFHub``),
not during checkpoint saving. Then override
:meth:`~xpm_torch.module.Module.loader_config` on your model to return
this loader:

.. code-block:: python

    class MyModel(Module):
        def loader_config(self, path):
            return MyCustomLoader.C(value=self, path=path)

Subclassing ``ExperimaestroHFHub``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For library-level customization (e.g. adding a README or TensorBoard
logs to every upload), subclass ``ExperimaestroHFHub``:

.. code-block:: python

    from experimaestro.huggingface import ExperimaestroHFHub

    class MyLibraryHFHub(ExperimaestroHFHub):
        def __init__(self, config, variant=None, readme=None):
            super().__init__(config, variant)
            self.readme = readme

        def _save_pretrained(self, save_directory):
            super()._save_pretrained(save_directory)
            if self.readme:
                (save_directory / "README.md").write_text(self.readme)

Utility functions
-----------------

Helper functions for working with HuggingFace Hub (cache checks,
downloads, config retrieval):

.. automodule:: xpm_torch.huggingface
   :members:
