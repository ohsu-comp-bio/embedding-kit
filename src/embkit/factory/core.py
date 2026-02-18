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

def load(path):
    state_dict = torch.load(path)
    desc = state_dict.pop("__model__", None)
    model = build(desc)
    model.load_state_dict(state_dict)
    return model