"""
Utility functions for working with PyTorch Lightning Fabric.
"""

import logging
from typing import Any, Optional, Union
import torch

logger = logging.getLogger(__name__)


def get_fabric_precision(fabric: Any) -> Optional[str]:
    """Extracts the precision string from a PyTorch Lightning Fabric instance.

    In PyTorch Lightning Fabric, precision settings are stored in `fabric._precision.precision`
    (e.g., 'bf16-mixed', '16-mixed', '32-true').

    Args:
        fabric: The PyTorch Lightning Fabric instance (or None).

    Returns:
        Optional[str]: The precision string if found, otherwise None.
    """
    if fabric is None:
        return None

    # Check PyTorch Lightning Fabric internal _precision plugin
    _precision_obj = getattr(fabric, "_precision", None)
    if _precision_obj is not None and hasattr(_precision_obj, "precision"):
        return str(_precision_obj.precision)

    # Fallback to direct attribute lookup if available
    p = getattr(fabric, "precision", None)
    if p is not None:
        return str(getattr(p, "precision", p))

    return None


def is_16bit_precision(precision: Optional[str]) -> bool:
    """Checks if the given PyTorch Lightning Fabric precision string represents 16-bit precision.

    Args:
        precision: The precision string from Fabric (e.g. '16-mixed', 'bf16-mixed', '32-true', None).

    Returns:
        bool: True if precision is a 16-bit precision format, False otherwise.
    """
    if precision is None:
        return False
    p = str(precision).lower()
    return p in ("16-mixed", "bf16-mixed", "16-true", "bf16-true", "16", "bf16")


def fallback_fa2_if_incompatible_precision(
    module: torch.nn.Module, fabric_or_precision: Union[Any, Optional[str]]
) -> None:
    """Checks if a module is configured for FlashAttention-2 under full (32-bit) precision and falls back to SDPA.

    FlashAttention-2 requires float16 or bfloat16 inputs. When Fabric runs in float32 precision
    (precision is None or '32-true'), this function updates any FlashAttention _attn_implementation
    attribute to 'sdpa' to prevent runtime errors.

    Args:
        module: The PyTorch module whose configurations will be inspected.
        fabric_or_precision: Either a PyTorch Lightning Fabric instance or a precision string.
    """
    if hasattr(fabric_or_precision, "_precision") or hasattr(
        fabric_or_precision, "precision"
    ):
        fabric_precision = get_fabric_precision(fabric_or_precision)
    else:
        fabric_precision = fabric_or_precision

    if is_16bit_precision(fabric_precision):
        return

    configs_to_check = []
    if hasattr(module, "config"):
        configs_to_check.append(module.config)
    if hasattr(module, "hf_model") and hasattr(module.hf_model, "config"):
        configs_to_check.append(module.hf_model.config)
    if (
        hasattr(module, "st_model")
        and hasattr(module.st_model, "model")
        and hasattr(module.st_model.model, "config")
    ):
        configs_to_check.append(module.st_model.model.config)
    if (
        hasattr(module, "encoder")
        and hasattr(module.encoder, "model")
        and hasattr(module.encoder.model, "config")
    ):
        configs_to_check.append(module.encoder.model.config)

    for cfg in configs_to_check:
        attn_impl = getattr(cfg, "_attn_implementation", None)
        if attn_impl and "flash" in str(attn_impl).lower():
            logger.warning(
                f"[Fabric Setup] Fabric precision is {fabric_precision!r} (float32). "
                f"FlashAttention-2 requires fp16 or bf16 precision. "
                f"Automatically falling back attention implementation from {attn_impl!r} to 'sdpa'."
            )
            cfg._attn_implementation = "sdpa"
