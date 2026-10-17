"""
Model build and (de)serialization core.

The factory is how embkit turns a *description* of a model — a JSON-style
structure of layers, activations, and constraints — into a live PyTorch model,
and back again. This is the mechanism that lets a trained embedding model be
saved to a single ``.model`` file and rebuilt on any machine without the code
that originally constructed it.

The three public entry points work together:

- :func:`build` — reconstruct a model object from a description.
- :func:`save` — serialize a model (weights + description) to disk.
- :func:`load` — the inverse of :func:`save`; the function you'll reach for
  most when loading a previously trained model.

A description is either a registered class instance (anything with ``to_dict``),
a plain dict carrying a ``"__class__"`` key, a list of such elements (built into
a :class:`~embkit.factory.mapping.Sequential`), or an activation name string.
"""
from .mapping import get_activation, Sequential
from .registry import CLASS_REGISTRY

from torch import nn
import torch

def build(desc):
    """
    Reconstruct a PyTorch model (or module) from a serializable description.

    This is the core of embkit's model (re)construction: it reads a description
    and, using the :data:`~embkit.factory.registry.CLASS_REGISTRY` populated by
    the :func:`~embkit.factory.registry.nn_module` decorator, resolves each
    ``"__class__"`` entry back to the original class and calls its ``from_dict``
    method. Nested structures are resolved recursively.

    The description may be any of:

    - an object exposing a ``to_dict`` method (normalized to a dict first);
    - a dict with a ``"__class__"`` key naming a registered class;
    - a list of descriptions, built into a
      :class:`~embkit.factory.mapping.Sequential` stack; or
    - a string naming a registered activation (e.g. ``"relu"``).

    Parameters:
        desc: A model/module description (see above).

    Returns:
        The reconstructed ``nn.Module`` instance (an ``nn.ModuleList``-style
        stack for lists, an activation instance for strings).

    Raises:
        TypeError: If ``desc`` is a dict naming an unregistered class, or a type
            that cannot be interpreted as a model description.

    Example:
        >>> from embkit import factory
        >>> model = factory.build({"__class__": "...Encoder", "units": 64})
    """
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
    state_dict = torch.load(path, map_location=device, weights_only=False)
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
