Overview
========

**xpm-torch** is a PyTorch training framework built on
`experimaestro <https://experimaestro-python.readthedocs.io/>`_ and
`Lightning Fabric <https://lightning.ai/docs/fabric/>`_. It bridges
experimaestro's configuration and experiment management system with
PyTorch model training, checkpointing, and HuggingFace Hub integration.

What's in the package
---------------------

Module system
~~~~~~~~~~~~~

The core abstraction is :class:`~xpm_torch.module.Module`, which combines
:class:`experimaestro.Config` (declarative parameters, hashing, serialization)
with ``torch.nn.Module`` (parameters, forward pass, device management).

- :class:`~xpm_torch.module.Module` — Base class for all models. Declares
  parameters with ``Param[T]``, initializes structure in
  :meth:`~xpm_torch.module.Module.__initialize__`, saves/loads weights with
  safetensors via :meth:`~xpm_torch.module.Module.save_model` /
  :meth:`~xpm_torch.module.Module.load_model`. Subclasses override
  :meth:`~xpm_torch.module.Module.loader_config` to control how the model
  is loaded from a checkpoint.

- :class:`~xpm_torch.module.ModuleLoader` — Lightweight task that initializes
  a model and loads its weights from a checkpoint directory. Produced by
  :meth:`Module.loader_config(path) <xpm_torch.module.Module.loader_config>`.

- :class:`~xpm_torch.module.ModuleContainer` — A plain ``nn.Module`` container
  that auto-detects which children have state and wraps them with Lightning
  Fabric via :meth:`~xpm_torch.module.ModuleContainer.setup_with_fabric`.

See :doc:`module` for the full API reference.

Training
~~~~~~~~

A complete training loop with checkpointing, validation, and
distributed training support:

- :class:`~xpm_torch.learner.Learner` — The main training task. Configures the
  model, optimizer(s), trainer, and validation listeners. The main loop runs in
  :meth:`~xpm_torch.learner.Learner.execute`.

- :class:`~xpm_torch.trainers.Trainer` / :class:`~xpm_torch.trainers.LossTrainer`
  — Defines how batches are produced and processed.

- :class:`~xpm_torch.trainers.context.TrainState` — Serializes epoch/step
  counters, model weights (safetensors), trainer state, and optimizer state
  via :meth:`~xpm_torch.trainers.context.TrainState.save` /
  :meth:`~xpm_torch.trainers.context.TrainState.load`.

- :class:`~xpm_torch.trainers.context.TrainerContext` — Passed through the
  training loop; holds the current state, TensorBoard writer, Fabric instance,
  and provides :meth:`~xpm_torch.trainers.context.TrainerContext.add_loss` /
  :meth:`~xpm_torch.trainers.context.TrainerContext.add_metric` for
  regularization and logging.

- :class:`~xpm_torch.learner.LearnerListener` — Hook called after each epoch
  (e.g. for validation and early stopping). Produces
  :class:`~xpm_torch.module.ModuleLoader` instances for the best checkpoint
  via :meth:`~xpm_torch.learner.LearnerListener.init_task`.

- :class:`~xpm_torch.module.ModuleLoader` carries optional
  :attr:`~xpm_torch.module.ModuleLoader.settings` for metadata
  (e.g. :class:`~xpm_torch.learner.CheckpointSettings` for epoch,
  :class:`~xpm_torch.validation.ValidationSettings` for validation key).

See :doc:`training` for the full API reference.

Optimization
~~~~~~~~~~~~

Configurable optimizers and schedulers:

- :class:`~xpm_torch.optim.ParameterOptimizer` — Associates an
  :class:`~xpm_torch.optim.Optimizer` (e.g.
  :class:`~xpm_torch.optim.Adam`,
  :class:`~xpm_torch.optim.AdamW`,
  :class:`~xpm_torch.optim.SGD`,
  :class:`~xpm_torch.optim.Adafactor`) with a
  :class:`~xpm_torch.schedulers.Scheduler` and optional
  :class:`~xpm_torch.optim.ParameterFilter` for per-group learning rates.

- :class:`~xpm_torch.optim.GradientClippingHook` /
  :class:`~xpm_torch.optim.GradientLogHook` — Hooks for gradient management.

See :doc:`optimization` for the full API reference.

Export actions
~~~~~~~~~~~~~~

After training, models can be exported to HuggingFace Hub or a local directory
via experimaestro's action system. The :class:`~xpm_torch.learner.Learner`
automatically registers :class:`~xpm_torch.actions.ExportAction` instances
during submission, which can be executed interactively after the experiment
completes. Subclass :class:`~xpm_torch.actions.ExportAction` and override
:meth:`~xpm_torch.module.Module.export_action` to customize the export
behavior for your models.

See :doc:`huggingface` for details on actions, pushing models, and customizing
the checkpoint format.

HuggingFace Hub
~~~~~~~~~~~~~~~

Utility functions for cache checking and downloading from HuggingFace Hub.
Model upload/download is handled by :class:`~xpm_torch.huggingface.TorchHFHub`
(which extends ``ExperimaestroHFHub`` from the experimaestro package)
to provide better integration for xpm-torch models and their loaders.

Experiment results
~~~~~~~~~~~~~~~~~~

- :class:`~xpm_torch.results.TrainingResults` — A serializable ``Config``
  holding trained model configs and TensorBoard log paths. Saved by experiments
  for later retrieval.

Batching
~~~~~~~~

- :class:`~xpm_torch.batchers.Batcher` /
  :class:`~xpm_torch.batchers.PowerAdaptativeBatcher` — Handles micro-batching
  with OOM recovery. Automatically reduces batch size on
  ``RecoverableOOMError`` and replays the failed batch.

Fabric configuration
~~~~~~~~~~~~~~~~~~~~

- :class:`~xpm_torch.configuration.FabricConfiguration` — Wraps Lightning
  Fabric settings (precision, devices, strategy, accelerator) as an
  experimaestro ``Config`` for declarative experiment setup.

How it fits together
--------------------

::

    YAML config
      → experimaestro deserializes Config objects
        → Learner task is submitted (ExportActions registered)
          → Fabric launches training
            → Trainer produces batches, Module computes forward/backward
              → TrainerContext collects losses and metrics
                → LearnerListeners validate and checkpoint
                  → ModuleLoader configs point to best checkpoints
      → After completion, ExportActions execute interactively
          → TorchHFHub serializes for Hub upload or local save

Related packages
----------------

- `experimaestro <https://experimaestro-python.readthedocs.io/>`_ —
  Configuration framework, task scheduling, workspace management
- `xpmir (experimaestro-IR) <https://experimaestro-ir.readthedocs.io/>`_ —
  Information Retrieval models and experiments built on xpm-torch
