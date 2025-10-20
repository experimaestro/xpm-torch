# xpm-torch

PyTorch utilities for experimaestro, providing seamless integration between PyTorch models and the experimaestro configuration framework.

## Installation

```bash
pip install xpm-torch
```

## xpmTorchHubModule

`xpmTorchHubModule` is a generic PyTorch module that combines experimaestro's configuration management with Hugging Face Hub compatibility via `ModelHubMixin`. This allows you to easily create configurable PyTorch models that can be saved, loaded, and shared on the Hugging Face Hub.

### Features

- **Configuration Management**: Leverages experimaestro's `Config` system for parameter management
- **Hugging Face Hub Integration**: Save and load models from Hugging Face Hub
- **Serialization**: Automatic serialization of model architecture and weights
- **Type Safety**: Uses typed parameters with experimaestro's `Param` system

### Basic Usage

#### 1. Define Your Model

Create a custom model by inheriting from `xpmTorchHubModule`:

```python
from xpm_torch.xpmModel import xpmTorchHubModule
from experimaestro import Param, Constant
import torch
import torch.nn as nn

class MyTorchModel(
    xpmTorchHubModule,
    library_name="my-org/my-model",
    tags=["example", "torch"],
    repo_url="https://github.com/my-org/my-model",
    paper_url="https://arxiv.org/abs/???"
):
    """Example model implementation"""
    
    # Define parameters
    input_dim: Param[int] = 100
    """Input dimension"""
    
    hidden_dim: Param[int] = 200
    """Hidden dimension"""
    
    output_dim: Param[int] = 10
    """Output dimension"""
    
    version: Constant[str] = "1.0"
    
    def __post_init__(self):
        # Initialize parent class
        super().__post_init__()
        
        # Define your layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

#### 2. Create and Use the Model

```python
# Create new model configuration
model_cfg = MyTorchModel.C(input_dim=50, hidden_dim=100, output_dim=5)

# Instantiate the model: we get a clean PyTorch Model
model = model_cfg.instance()

# Use the model
input_tensor = torch.randn(1, 50)
output = model(input_tensor)
```

#### 3. Save Your Model

**Save locally:**

```python
# Save to local directory
model.save_pretrained("my_model_dir")
```

**Push to Hugging Face Hub:**

```python
# Push to HF Hub
model.push_to_hub("my-username/my-model-id")
```

#### 4. Load Your Model

**Load from local directory:**

```python
loaded_model = MyTorchModel.from_pretrained("my_model_dir")
```

**Load from Hugging Face Hub:**

```python
loaded_model = MyTorchModel.from_pretrained(
    "my-username/my-model-id",
    force_download=True
)
```

### Alternative Constructor

You can also use `from_kwargs` to create a model directly from keyword arguments:

```python
model = MyTorchModel.from_kwargs(input_dim=50, hidden_dim=100, output_dim=5)
```

### Model Properties

#### `device`
Get the device where the model is located:

```python
device = model.device
```

#### `count_parameters()`
Count the number of trainable parameters:

```python
num_params = model.count_parameters()
print(f"Model has {num_params} trainable parameters")
```

### How It Works

1. **Configuration**: When you define parameters using `Param`, experimaestro tracks them as configuration options
2. **Serialization**: `save_pretrained()` serializes both the model weights (as a PyTorch state dict) and the configuration
3. **Loading**: `from_pretrained()` deserializes the configuration, instantiates the model, and loads the weights
4. **Hub Integration**: The `ModelHubMixin` handles uploading/downloading from Hugging Face Hub

### Advanced Features

#### Custom Metadata

You can add custom metadata to your model class:

```python
class MyModel(
    xpmTorchHubModule,
    library_name="my-org/my-model",
    tags=["nlp", "classification", "torch"],
    repo_url="https://github.com/my-org/my-model",
    paper_url="https://arxiv.org/abs/2301.12345"
):
    # ... model definition
```

#### Using Constants

Use `Constant` for values that shouldn't change after instantiation:

```python
version: Constant[str] = "1.0"
model_type: Constant[str] = "encoder"
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
