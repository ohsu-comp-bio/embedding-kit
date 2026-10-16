from .mapping import get_activation, Sequential
from .registry import CLASS_REGISTRY

from torch import nn
import torch

def build(desc):
    if getattr(desc, "to_dict", None) is not None:
        desc = desc.to_dict()

    if isinstance(desc, dict):
        class_name = desc["__class__"]
        if class_name in CLASS_REGISTRY:
            return CLASS_REGISTRY[class_name].from_dict(desc)
        raise TypeError(f"Unknown layer type: {class_name}")
    elif isinstance(desc, list):
        elements = []
        for element in desc:
            elements.append(build(element))
        return Sequential(*elements)
    elif isinstance(desc, str):
        cls = get_activation(desc)
        if cls is not None:
            return cls()

    raise TypeError(f"Invalid input for build function: {type(desc)}")

def save(model, path):
    """Serialize a model and its description to ``path``.

    Clamps any masked weights before saving, writes the state dict along with
    the ``__model__`` descriptor, and persists everything via ``torch.save``.

    Args:
        model: The model to serialize.
        path: File path where the serialized model will be saved.
    """
    with torch.no_grad():
        for module in model.modules():
            clamp = getattr(module, "clamp_masked_weights", None)
            if callable(clamp):
                clamp()
    state = model.state_dict()
    desc = model.to_dict()    
    state["__model__"] = desc
    torch.save(state, path)

def load(path, device=None, dtype=None):
    """Load a serialized model from ``path`` and optionally move its tensors."""

    def patch_legacy_linear():
        '''Check for final linear layer expected in legacy loading and patch if present.'''
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            return False
        idx = len(encoder.net)
        weight_key, bias_key = f"encoder.net.{idx}.weight", f"encoder.net.{idx}.bias"
        if weight_key not in result.unexpected_keys or bias_key not in result.unexpected_keys:
            return False
        from .mapping import Linear
        out_features, in_features = state_dict[weight_key].shape
        encoder.net.append(Linear(in_features, out_features))
        return True

    state_dict = torch.load(path, map_location=device, weights_only=False)
    desc = state_dict.pop("__model__", None)
    if desc is None:
        raise KeyError(
            "Missing '__model__' key in the loaded state dict. "
            "The file does not contain a model description and cannot be loaded."
        )
    model = build(desc)
    result = model.load_state_dict(state_dict, strict = False) # Load non-strict

    if result.unexpected_keys and patch_legacy_linear():
        model.load_state_dict(state_dict, strict = True) # Legacy matches, Strict loading
    elif result.unexpected_keys:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model.__class__.__name__}: "
            f"Unexpected key(s) in state_dict: {result.unexpected_keys}."
        )
    elif result.missing_keys:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model.__class__.__name__}: "
            f"Missing key(s) in state_dict: {result.missing_keys}."
        )

    if device is not None or dtype is not None:
        model.to(device=device, dtype=dtype)
    return model

def run_model_verification(model_path, device=None):
    """
    Load a model and run its integrity verification logic.

    Args:
        model_path: Path to the .model file.
        device: Device to load the model on.

    Returns:
        A dictionary containing the verification report.
    """
    model = load(model_path, device=device)
    if hasattr(model, "verify_integrity"):
        report = model.verify_integrity()
    else:
        # Fallback for models that don't implement the interface yet
        report = {
            "model_type": model.__class__.__name__,
            "healthy": True,
            "issues": ["Model does not implement verify_integrity; fallback checks only."],
            "fallback_audit": True
        }
        # Basic NaN/Inf check
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                report["healthy"] = False
                report["issues"].append(f"NaN values detected in parameter: {name}")
            if torch.isinf(param).any():
                report["healthy"] = False
                report["issues"].append(f"Infinite values detected in parameter: {name}")

    # Attach lightweight identity metadata when available.
    if getattr(model, "features", None) is not None:
        report["feature_names"] = list(model.features)
        report.setdefault("features_count", len(model.features))
    if getattr(model, "latent_dim", None) is not None:
        report["declared_latent_dim"] = int(model.latent_dim)

    return report
