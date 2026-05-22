Training
========

Learner
-------

.. autoxpmconfig:: xpm_torch.learner.Learner

.. autoxpmconfig:: xpm_torch.learner.LearnerListener

.. autoxpmconfig:: xpm_torch.learner.CheckpointSettings

.. autoxpmconfig:: xpm_torch.learner.Strategy

Trainers
--------

.. autoxpmconfig:: xpm_torch.trainers.Trainer

.. autoxpmconfig:: xpm_torch.trainers.LossTrainer

.. autoxpmconfig:: xpm_torch.trainers.multiple.MultipleTrainer

Training State
--------------

.. autoclass:: xpm_torch.trainers.context.TrainState
   :members:

.. autoclass:: xpm_torch.trainers.context.TrainerContext
   :members:

Training Hooks
--------------

.. autoxpmconfig:: xpm_torch.trainers.context.TrainingHook

.. autoxpmconfig:: xpm_torch.trainers.context.StepTrainingHook

.. autoxpmconfig:: xpm_torch.trainers.context.ValidationHook

.. autoxpmconfig:: xpm_torch.trainers.context.InitializationTrainingHook

.. autoxpmconfig:: xpm_torch.trainers.hooks.LayerFreezer

.. autoxpmconfig:: xpm_torch.trainers.hooks.LayerSharer

Validation
----------

.. autoxpmconfig:: xpm_torch.validation.ValidationSettings

.. autoxpmconfig:: xpm_torch.trainers.validation.TrainerValidationLoss

Batching
--------

A ``Batcher`` controls how an outer batch is split into microbatches before
each forward/backward. Three strategies are available; the right choice
depends on whether you train on a single GPU or with multi-GPU (DDP), and on
whether your per-batch memory cost varies (e.g. variable sequence length in
transformers).

* ``Batcher`` — base, no-op. Uses the configured ``batch_size`` directly,
  no microbatching, no fallback. Good when you have already manually picked a
  microbatch size that always fits.

* ``PowerAdaptativeBatcher`` — single-GPU only. Catches CUDA OOM errors and
  shrinks the microbatch (``batch_size // n`` for increasing ``n``) until the
  call succeeds. **Do not use under DDP**: when one rank shrinks its
  microbatch and the others don't, the ranks desynchronize at the next NCCL
  collective and the job hangs (see
  https://github.com/pytorch/pytorch/issues/50820).

* ``PredictiveBatcher`` — fits a user-supplied memory cost formula (e.g.
  transformer attention: ``a*bs*l**2 + b*bs*l + c``) from a small set of
  probe batches run **before DDP wrapping**, then chooses the largest safe
  microbatch per outer batch using the fitted model. Avoids OOM-driven
  recovery entirely, so DDP cannot desynchronize from per-rank shrinkage.
  Recommended whenever you train on more than one GPU.

.. autoxpmconfig:: xpm_torch.batchers.Batcher

.. autoxpmconfig:: xpm_torch.batchers.PowerAdaptativeBatcher

.. autoxpmconfig:: xpm_torch.batchers.PredictiveBatcher

.. autoxpmconfig:: xpm_torch.batchers.BatchDimsProvider

.. autoxpmconfig:: xpm_torch.batchers.ProbePoint

Microbatch packing
~~~~~~~~~~~~~~~~~~

``PredictiveBatcher`` delegates the "how to split an outer batch into
microbatches" decision to a swappable ``MicroBatchPacker``. Two packers ship
with xpm-torch:

* ``UniformPacker`` (default) — compute dims once from the outer batch, choose
  a single microbatch size, slice equally. Optimal when records are
  homogeneous; conservative when they aren't.
* ``AdaptivePacker`` — accumulate records into a microbatch incrementally,
  emit when adding one more would exceed the budget. Optionally sort by a
  per-record cost proxy descending (``sort_by``) so heavy records go in one
  early microbatch and light records pack densely into later ones.

Use ``AdaptivePacker`` when records have heterogeneous cost (e.g. variable
sequence length) — each microbatch is then sized for its actual contents
rather than the worst-case record in the outer batch.

.. code-block:: python

    from xpm_torch.batchers import AdaptivePacker, PredictiveBatcher

    PredictiveBatcher.C(
        formula="a*bs*l**2 + b*bs*l + c",
        coefficient_names={"a", "b", "c"},
        variable_names={"bs", "l"},
        dims_provider=TextBatchDimsProvider.C(probe_dims=...),
        packer=AdaptivePacker.C(sort_by="l"),
    )

.. autoxpmconfig:: xpm_torch.batchers.MicroBatchPacker

.. autoxpmconfig:: xpm_torch.batchers.UniformPacker

.. autoxpmconfig:: xpm_torch.batchers.AdaptivePacker

Predictive batching: worked example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a transformer-style model where memory scales with batch size and the
square of sequence length, you would:

1. Write a ``BatchDimsProvider`` for your record type, exposing the batch size
   and max sequence length and the set of probe points.
2. Plug it into a ``PredictiveBatcher`` and pass that to your trainer's
   ``batcher`` parameter.

.. code-block:: python

    from experimaestro import Param, sealed_set
    from xpm_torch.batchers import (
        BatchDimsProvider,
        PredictiveBatcher,
        ProbePoint,
    )


    class TextBatchDimsProvider(BatchDimsProvider):
        """Reads (bs, l) from a list-of-records batch where each record has a
        ``tokens`` attribute."""

        def compute_batch_dims(self, batch) -> dict[str, int]:
            return {
                "bs": len(batch),
                "l": max(len(r.tokens) for r in batch),
            }

        def build_probe_batch(self, dims: dict[str, int]):
            # Return a synthetic batch matching the requested dims. Replace
            # this with your project's record constructor.
            return [DummyRecord(tokens=[0] * dims["l"]) for _ in range(dims["bs"])]


    trainer = LossTrainer.C(
        batcher=PredictiveBatcher.C(
            formula="a*bs*l**2 + b*bs*l + c",
            coefficient_names={"a", "b", "c"},
            variable_names={"bs", "l"},
            batch_size_variable="bs",
            dims_provider=TextBatchDimsProvider.C(
                probe_dims=sealed_set(
                    ProbePoint.C(values={"bs": 1, "l": 64}),
                    ProbePoint.C(values={"bs": 1, "l": 128}),
                    ProbePoint.C(values={"bs": 2, "l": 128}),
                    ProbePoint.C(values={"bs": 4, "l": 64}),
                    ProbePoint.C(values={"bs": 2, "l": 256}),
                ),
            ),
        ),
        batch_size=32,
        sampler=...,
    )

Notes:

* The ``formula`` is canonicalized in ``__validate__`` (terms sorted, whitespace
  normalized) so equivalent inputs share an experimaestro task identifier.
* Probes run on rank 0 only; the fitted coefficients are broadcast to other
  ranks. This is safe because probing happens before
  ``fabric.setup(model)`` — a rank-0-only forward+backward on a DDP-wrapped
  model would deadlock at the first gradient bucket all-reduce.
* The default ``memory_fraction=0.75`` is conservative to absorb DDP gradient
  buckets and NCCL communication buffers that the pre-DDP probe doesn't see.
  If you still OOM at train time, prefer lowering ``memory_fraction`` or
  setting ``extra_overhead_bytes`` over silently retrying — silent retry
  re-introduces the very NCCL desync risk this batcher exists to avoid.

Batchwise Losses
~~~~~~~~~~~~~~~~

.. autoxpmconfig:: xpm_torch.losses.batchwise.BatchwiseLoss

.. autoxpmconfig:: xpm_torch.losses.batchwise.CrossEntropyLoss

.. autoxpmconfig:: xpm_torch.losses.batchwise.SoftmaxCrossEntropy

Pairwise Losses
~~~~~~~~~~~~~~~

.. autoxpmconfig:: xpm_torch.losses.pairwise.PairwiseLoss

.. autoxpmconfig:: xpm_torch.losses.pairwise.CrossEntropyLoss

.. autoxpmconfig:: xpm_torch.losses.pairwise.HingeLoss

.. autoxpmconfig:: xpm_torch.losses.pairwise.PointwiseCrossEntropyLoss

Fabric Configuration
--------------------

.. autoxpmconfig:: xpm_torch.configuration.Strategy

.. autoxpmconfig:: xpm_torch.configuration.FabricConfigurationBase

.. autoxpmconfig:: xpm_torch.configuration.FabricConfiguration
