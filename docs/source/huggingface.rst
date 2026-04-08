HuggingFace Hub Integration
===========================

xpm-torch models can be pushed to and loaded from the
`HuggingFace Hub <https://huggingface.co/>`_ via
:class:`~xpm_torch.huggingface.TorchHFHub` (which extends experimaestro's
``ExperimaestroHFHub``). The serialization preserves the full experimaestro
configuration graph, so a downloaded model can be used directly in further
experiments.

Exporting models via actions
----------------------------

The recommended way to export trained models is through experimaestro's
**action system**. When a :class:`~xpm_torch.learner.Learner` is submitted,
it automatically registers :class:`~xpm_torch.actions.ExportAction` instances
for the last checkpoint and for each listener's best checkpoint. After the
experiment completes, these actions can be executed interactively via the
experimaestro CLI (``experimaestro experiments actions``).

:class:`~xpm_torch.actions.ExportAction` prompts the user to choose between
uploading to HuggingFace Hub or saving to a local directory, then delegates
to :class:`~xpm_torch.huggingface.TorchHFHub`.

How actions are registered
~~~~~~~~~~~~~~~~~~~~~~~~~~

Actions are registered during task submission via the ``add_action`` callback
provided by experimaestro:

.. code-block:: python

    class Learner(Task):
        def __submit__(self, dep, add_action):
            loader = dep(self.model.loader_config(...))
            # Register export action for the last checkpoint
            add_action(self.model.export_action(loader, default_name="last"))
            ...

:meth:`Module.export_action(loader, **kwargs) <xpm_torch.module.Module.export_action>`
returns an :class:`~xpm_torch.actions.ExportAction` config by default.
Subclasses override this method to return library-specific actions (e.g.
``XPMIRExportAction`` adds xpmir README sections and metadata).

Customizing the export action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To customize what happens during export, subclass
:class:`~xpm_torch.actions.ExportAction` and override
:meth:`~xpm_torch.actions.ExportAction.get_hub` to return a
library-specific hub wrapper:

.. code-block:: python

    from xpm_torch.actions import ExportAction

    class MyExportAction(ExportAction):
        def get_hub(self):
            return MyCustomHFHub(self.loader)

Then override :meth:`~xpm_torch.module.Module.export_action` on your model
to return this action:

.. code-block:: python

    class MyModel(Module):
        def export_action(self, loader, **kwargs):
            return MyExportAction.C(loader=loader, **kwargs)

Direct API usage
~~~~~~~~~~~~~~~~

You can also use :class:`~xpm_torch.huggingface.TorchHFHub` directly:

.. code-block:: python

    from xpm_torch.huggingface import TorchHFHub

    # Push a ModuleLoader (from loader_config or validation output)
    TorchHFHub(loader).push_to_hub("your-org/model-name")

    # Or save locally first
    TorchHFHub(loader).save_pretrained("/path/to/save")

:class:`~xpm_torch.huggingface.TorchHFHub` calls
:meth:`~xpm_torch.module.ModuleLoader.write_hub_extras` and
:meth:`~xpm_torch.module.ModuleLoader.hub_readme_sections` on the loader,
so format-specific files (e.g. sentence-transformers configs) and README
sections are generated automatically.

What gets uploaded
~~~~~~~~~~~~~~~~~~

The serialized directory contains:

- ``experimaestro.json`` — the config definition (for reloading with xpmir)
- Model weight directories — named after the loader's ``DataPath`` fields
  (or customized via ``__xpm_serialize__``)
- Format-specific configs written by ``write_hub_extras`` (e.g.
  ``modules.json``, ``router_config.json`` for sentence-transformers)
- ``README.md`` — assembled from base + loader sections

Loading a model from the Hub
-----------------------------

You can load models from the HuggingFace Hub using :class:`~xpm_torch.huggingface.TorchHFHub`.
There are two main ways to load a model depending on whether you want:
- Direct access to the initialized model instance 
- Only configuration (loader) itself (for instance in an experiment file).

Loading a model instance
~~~~~~~~~~~~~~~~~~~~~~~~

To load a model as a ready-to-use instance (already initialized and with
weights loaded), use :meth:`~xpm_torch.huggingface.TorchHFHub.from_pretrained`.
This is ideal for direct inference or when you don't need to manipulate
the configuration:

.. code-block:: python

    from xpm_torch.huggingface import TorchHFHub

    # Returns an initialized Module instance with weights loaded
    model = TorchHFHub.from_pretrained("your-org/model-name")

Loading a model loader
~~~~~~~~~~~~~~~~~~~~~~

To load a :class:`~xpm_torch.module.ModuleLoader` configuration instead of
an instance, use :meth:`~xpm_torch.huggingface.TorchHFHub.pretrained_loader`.
This returns a ``ModuleLoader`` config object that can be used as an
initialization task in larger experiments:

.. code-block:: python

    from xpm_torch.huggingface import TorchHFHub

    # Returns a ModuleLoader config object
    loader_cfg = TorchHFHub.pretrained_loader("your-org/model-name")

    # The model config is accessible via loader_cfg.model
    model_cfg = loader_cfg.model

Low-level access
~~~~~~~~~~~~~~~~

If you need the raw deserialized data from the Hub without any xpm-torch
specific processing, you can use the base experimaestro class:

.. code-block:: python

    from experimaestro.huggingface import ExperimaestroHFHub

    # Returns the deserialized config (e.g. a ModuleLoader)
    data = ExperimaestroHFHub.from_pretrained("your-org/model-name")

Customizing the HF checkpoint format
-------------------------------------

There are three extension points for customizing what gets written during
Hub export:

- :meth:`Module.loader_config(path) <xpm_torch.module.Module.loader_config>` —
  on the model, controls which ``ModuleLoader`` subclass is returned
- :meth:`ModuleLoader.write_hub_extras(save_directory) <xpm_torch.module.ModuleLoader.write_hub_extras>` —
  on the loader, writes additional files (e.g. ST configs)
- :meth:`ModuleLoader.hub_readme_sections() <xpm_torch.module.ModuleLoader.hub_readme_sections>` —
  on the loader, provides named README sections with positioning

The hooks are on :class:`~xpm_torch.module.ModuleLoader` (not on
:class:`~xpm_torch.module.Module`) because the loader is the object that
gets serialized for Hub export and holds the ``DataPath`` references to
model weights. ``Module`` configs are data-less.

:meth:`~xpm_torch.module.Module.loader_config`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a model is serialized for the Hub, experimaestro serializes the
:class:`~xpm_torch.module.ModuleLoader` returned by
:meth:`~xpm_torch.module.Module.loader_config`. Subclasses override this
method to return a custom loader with different ``DataPath`` fields:

.. code-block:: python

    from xpm_torch.module import Module, SimpleModuleLoader

    class MyModel(Module):
        def loader_config(self, path):
            # Default: single path DataPath
            return SimpleModuleLoader.C(value=self, path=path)

Loaders can also override ``__xpm_serialize__`` to control the directory
names used during serialization (e.g. mapping field names to
sentence-transformers conventions as done [here](https://github.com/experimaestro/experimaestro-ir/blob/d685910db9222e7b4b95aaf30e94d6052f27c6f8/src/xpmir/neural/splade.py#L235)).

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

These hooks are only called during Hub export (by
:class:`~xpm_torch.huggingface.TorchHFHub`), not during checkpoint saving.
Then override :meth:`~xpm_torch.module.Module.loader_config` on your model
to return this loader:

.. code-block:: python

    class MyModel(Module):
        def loader_config(self, path):
            return MyCustomLoader.C(value=self, path=path)

Class hierarchy
~~~~~~~~~~~~~~~

- :class:`~experimaestro.huggingface.ExperimaestroHFHub` — base serialization
  (experimaestro)
- :class:`~xpm_torch.huggingface.TorchHFHub` — calls ``write_hub_extras``
  and ``hub_readme_sections`` on the loader (xpm-torch)
- ``XPMIRHFHub`` — adds xpmir README sections (frontmatter, usage, results)
  and TensorBoard logs (xpmir)

Utility functions
-----------------

Helper functions and classes for working with HuggingFace Hub:

.. automodule:: xpm_torch.huggingface
   :members:
