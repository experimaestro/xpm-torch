Training
========

Learner
-------

.. autoxpmconfig:: xpm_torch.learner.Learner

.. autoxpmconfig:: xpm_torch.learner.LearnerListener

.. autoxpmconfig:: xpm_torch.learner.CheckpointModuleLoader

.. autoxpmconfig:: xpm_torch.learner.Strategy

Trainers
--------

.. autoxpmconfig:: xpm_torch.trainers.Trainer

.. autoxpmconfig:: xpm_torch.trainers.LossTrainer

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

.. autoxpmconfig:: xpm_torch.validation.ValidationModuleLoader

Batching
--------

.. autoxpmconfig:: xpm_torch.batchers.Batcher

.. autoxpmconfig:: xpm_torch.batchers.PowerAdaptativeBatcher

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
