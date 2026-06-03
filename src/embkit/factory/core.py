from .mapping import get_activation, Sequential
from .registry import CLASS_REGISTRY

from torch import nn
import torch

def build(desc):
    if getattr(desc, "to_dict", None) is not None:
        desc = desc.to_dict()

    if isinstance(desc, dict):
        className = desc["__class__"]
        if className not in CLASS_REGISTRY:
            try:
                from .base_vae import _import_obj
                _import_obj(className)
            except Exception as e:
                pass
        
        if className in CLASS_REGISTRY:
            return CLASS_REGISTRY[className].from_dict(desc)
        raise Exception(f"Unknown layer type: {className}")
    elif isinstance(desc, list):
        elements = []
        for element in desc:
            elements.append(build(element))
        return Sequential(*elements)
    elif isinstance(desc, str):
        cls = get_activation(desc)
        if cls is not None:
            return cls()

    raise Exception(f"Invalid input for build function: {type(desc)}")

def save(model, path):
    # Safety net: clamp constrained weights before serialization.
    if isinstance(model, nn.Module):
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
    state_dict = torch.load(path, map_location=device)
    desc = state_dict.pop("__model__", None)
    if desc is None:
        raise KeyError(
            "Missing '__model__' key in the loaded state dict. "
            "The file does not contain a model description and cannot be loaded."
        )
    model = build(desc)
    model.load_state_dict(state_dict)
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
