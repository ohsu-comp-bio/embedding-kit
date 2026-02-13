


# In a module like `registry.py`
from typing import Dict, Type

# The global map (registry) where classes will be stored
CLASS_REGISTRY: Dict[str, Type] = {}

def register_nn_module(cls: Type) -> Type:
    """A class decorator to register classes in the global registry."""
    # Register the class using its name as the key
    CLASS_REGISTRY[cls.__name__] = cls
    # Return the class object unmodified
    return cls
