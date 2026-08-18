import torch
from torch.optim import SGD
from xpm_torch.schedulers import LinearWithWarmupAndMaxEpochs


def test_linear_with_warmup_and_max_epochs():
    param = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = SGD([param], lr=1.0)

    # total_training_steps = 35000, num_warmup_steps = 1000
    scheduler_config = LinearWithWarmupAndMaxEpochs.C(
        num_warmup_steps=1000,
        total_training_steps=35000,
    )

    # Instantiate scheduler via __call__ with num_training_steps=15000 (Phase 1 default)
    scheduler = scheduler_config(optimizer, num_training_steps=15000)

    # Test step 0 (warmup start)
    assert scheduler.get_last_lr()[0] == 0.0

    # Test step 1000 (warmup end)
    for _ in range(1000):
        scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - 1.0) < 1e-4

    # Fast forward to step 15000 (end of Phase 1)
    for _ in range(14000):
        scheduler.step()
    expected_lr_15k = (35000 - 15000) / (35000 - 1000)  # 20000 / 34000 ≈ 0.588235
    assert abs(scheduler.get_last_lr()[0] - expected_lr_15k) < 1e-4

    # Fast forward to step 35000 (end of Phase 2)
    for _ in range(20000):
        scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - 0.0) < 1e-4
