

# In a module like `registry.py`
from typing import Dict, Type

# The global map (registry) where classes will be stored
CLASS_REGISTRY: Dict[str, Type] = {}

def nn_module(cls: Type) -> Type:
    """A class decorator to register classes in the global registry."""
    name = get_class_name(cls)
    return nn_module_wrap_register(cls, name)

class nn_module_named:
    def __init__(self, name):
        self.name = name
    def __call__(self, cls):
        return nn_module_wrap_register(cls, self.name)

def class_dict_wrapper(base_class):
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
    """A class decorator to register classes with given name"""
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
        result['__class__'] = get_class_name(cls)
        return result

    # Replace the original to_dict method with the wrapper
    cls.to_dict = to_dict_wrapper
    return cls

def get_class_name(cls):
    """Inspect a class to get standard name"""
    return cls.__module__ + "." + cls.__name__
