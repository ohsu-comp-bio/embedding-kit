from .mapping import get_activation, Sequential
from .registery import CLASS_REGISTRY

from torch import nn


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
