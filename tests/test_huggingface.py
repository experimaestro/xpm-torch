import torch
import torch.nn as nn
from experimaestro import Param
from xpm_torch.module import Module, SimpleModuleLoader
from xpm_torch.huggingface import TorchHFHub
from pathlib import Path


class DummyModule(Module):
    dim: Param[int]

    def __initialize__(self):
        self.linear = nn.Linear(self.dim, 1)

    def forward(self, x):
        return self.linear(x)


def test_torch_hf_hub_loading(tmp_path):
    """Test TorchHFHub from_pretrained and pretrained_loader."""
    # 1. Setup: Create and save a model locally
    cfg = DummyModule.C(dim=10)
    model = cfg.instance()
    model.initialize()
    
    # Run forward to get a reference output
    x = torch.randn(1, 10)
    expected_out = model(x)

    # Save weights
    model_dir = tmp_path / "model_weights"
    model.save_model(model_dir)

    # Create a loader for this model
    loader = SimpleModuleLoader.C(value=cfg, path=model_dir)
    
    # Save as a "pretrained" model (mimicking Hub structure)
    save_dir = tmp_path / "pretrained_hub"
    hub = TorchHFHub(loader)
    hub.save_pretrained(save_dir)

    # 2. Test TorchHFHub.from_pretrained -> should return the initialized Module
    loaded_model = TorchHFHub.from_pretrained(save_dir)
    assert isinstance(loaded_model, DummyModule)
    assert loaded_model._initialized
    
    # Verify weights were loaded correctly
    actual_out = loaded_model(x)
    assert torch.allclose(expected_out, actual_out)

    # 4. Test TorchHFHub.pretrained_loader with as_instance=True
    loaded_loader_instance = TorchHFHub.pretrained_loader(save_dir, as_instance=True)
    assert isinstance(loaded_loader_instance, SimpleModuleLoader)
    # Since it's an instance, we should be able to call execute directly (or it might already be executed?)
    # Usually in experimaestro, as_instance=True means it is deserialized AND instantiated.
    # It doesn't necessarily mean execute() was called.
    loaded_loader_instance.execute()
    assert torch.allclose(expected_out, loaded_loader_instance.model(x))
