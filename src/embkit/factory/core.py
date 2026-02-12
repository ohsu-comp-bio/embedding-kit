
from .mapping import classMap, convert_activation

from torch import nn

def build(desc):

    if getattr(desc, "to_dict", None) is not None:
        desc = desc.to_dict()

    if isinstance(desc, dict):
        className = desc["__class__"]
        if className in classMap:
            return classMap[className].from_dict(desc)
        raise Exception(f"Unknown layer type: {className}")
    elif isinstance(desc, list):
        elements = []
        for element in desc:
            elements.append(build(element))
        return nn.Sequential(*elements)
    elif isinstance(desc, str):
        cls = convert_activation(desc)
        if cls is not None:
            return cls

    raise Exception(f"Invalid input for build function: {type(desc)}")