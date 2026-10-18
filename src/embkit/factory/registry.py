
"""
Class registry for serializable model components.

This is the piece that makes :func:`~embkit.factory.core.build` possible. The
factory must turn a saved model description back into real objects, and to do
that it needs a name-to-class lookup table. The registry is that table: classes
opt in by being decorated with :func:`nn_module` (or :func:`nn_module_named`),
which records them under a stable, fully-qualified name and requires them to
implement the ``to_dict``/``from_dict`` contract.

The registry is what decouples a model description from the exact import path
that built it — the same descriptor can be rebuilt long after the original
training script is gone.
"""
from typing import Dict, Type

# The global map (registry) where classes will be stored
CLASS_REGISTRY: Dict[str, Type] = {}

def get_class_name(cls):
    """Inspect a class to get standard name"""
    return cls.__module__ + "." + cls.__name__

def nn_module(cls: Type) -> Type:
    """A class decorator to register classes in the global registry."""
    name = get_class_name(cls)
    return nn_module_wrap_register(cls, name)

class nn_module_named:
    """
    Register a class under an explicit name, instead of its fully-qualified one.

    This is the named form of the :func:`nn_module` decorator. By default a class
    is registered under ``module.ClassName``; when you want a model description to
    reference a class by a specific, stable identifier (e.g. to keep a name stable
    across refactors, or to register the same class under an alias), pass that
    name here instead.

    Like :func:`nn_module`, the decorated class must implement both ``to_dict``
    and ``from_dict`` — it is rejected otherwise.

    Example:
        @nn_module_named("embkit.factory.LegendaryLayer")
        class LegendaryLayer:
            ...
    """
    def __init__(self, name):
        self.name = name
    def __call__(self, cls):
        return nn_module_wrap_register(cls, self.name)

def class_dict_wrapper(base_class):
    """
    A class decorator to wrap a class with to_dict and from_dict methods.
    Note: to primarily designed for classes with no init args. And is used for a 
    limited set of classes
    """
    new_name = f"embkit.factory.mapping.{base_class.__name__}"

    def to_dict(self):
        return {
            "__class__": new_name,
            "params": {} # Empty since you specified no init args
        }

    @classmethod
    def from_dict(cls, data):
        # Simply instantiate the class
        return cls()

    # 3. Create a new class dynamically
    # type(name, bases, dict)
    new_class = type(new_name, (base_class,), {
        "to_dict": to_dict,
        "from_dict": from_dict
    })
    CLASS_REGISTRY[new_name] = new_class
    return new_class


def nn_module_wrap_register(cls: Type, name: str) -> Type:
    """
    Internal method for class regiostration. Use nn_module or nn_module_named instead of calling this directly.
    """
    if not hasattr(cls, "to_dict") or not hasattr(cls, "from_dict"):
        raise ValueError("Class must have both `to_dict` and `from_dict` methods.")
    CLASS_REGISTRY[name] = cls

    # Store the original to_dict method
    original_to_dict = cls.to_dict
    # Define a wrapper method that adds class info to the dictionary
    def to_dict_wrapper(self):
        # Call the original to_dict method
        result = original_to_dict(self)
        # Add class information to the result
        result['__class__'] = name
        return result

    # Replace the original to_dict method with the wrapper
    cls.to_dict = to_dict_wrapper
    return cls
