


# In a module like `registry.py`
from typing import Dict, Type

# The global map (registry) where classes will be stored
CLASS_REGISTRY: Dict[str, Type] = {}

def nn_module(cls: Type) -> Type:
    """A class decorator to register classes in the global registry."""
    if not hasattr(cls, "to_dict") or not hasattr(cls, "from_dict"):
        raise ValueError("Class must have both `to_dict` and `from_dict` methods.")
    # Register the class using its name as the key
    CLASS_REGISTRY[get_class_name(cls)] = cls
    # Return the class object unmodified
    return cls

def nn_module_named(cls: Type, name: str) -> Type:
    """A class decorator to register classes with given name"""
    if not hasattr(cls, "to_dict") or not hasattr(cls, "from_dict"):
        raise ValueError("Class must have both `to_dict` and `from_dict` methods.")
    CLASS_REGISTRY[name] = cls
    return cls

def get_class_name(cls):
    """Inspect a class to get standard name"""
    return cls.__module__ + "." + cls.__name__
