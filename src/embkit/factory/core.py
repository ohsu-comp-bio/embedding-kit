from .mapping import get_activation, Sequential
from .registry import CLASS_REGISTRY

from torch import nn
import torch

def build(desc):
    if getattr(desc, "to_dict", None) is not None:
        desc = desc.to_dict()

    if isinstance(desc, dict):
        className = desc["__class__"]
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