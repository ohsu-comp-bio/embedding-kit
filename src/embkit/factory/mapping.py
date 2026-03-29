
"""
Docstring for embkit.factory.mapping

Lightweight mappings for nn module, designed to 
provide a consistent way to do to_dict and from_dict methods for nn modules.
"""

from typing import Optional, Type

from torch import nn

from .registry import nn_module, get_class_name, class_dict_wrapper, CLASS_REGISTRY

def clean_params(params):
    out = {}
    for k, v in params.items():
        if k not in ["__class__"]:
            out[k] = v
    return out

@nn_module
class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        self._params = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "device": device,
            "dtype": dtype
        }
    
    @classmethod
    def from_dict(cls, params):
        return cls(**clean_params(params))
    
    def to_dict(self):
        return self._params

@nn_module
class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, device=None, dtype=None):
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, device=device, dtype=dtype)
        self._params = {
            "num_features": num_features,
            "eps": eps,
            "momentum": momentum,
            "affine": affine,
            "device": device,
            "dtype": dtype
        }

    @classmethod
    def from_dict(cls, params):
        return cls(**clean_params(params))
    
    def to_dict(self):
        return self._params

@nn_module
class Sequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self._params = {"args": args}
    
    @classmethod
    def from_dict(cls, params):
        args = params["args"]
        modules = []
        for a in args:
            modules.append( CLASS_REGISTRY[a["__class__"]].from_dict(a ) )
        return cls(*modules)
    
    def to_dict(self):
        args = self._params["args"]
        out = []
        for a in args:
            out.append( a.to_dict() )
        return { "args": out }


ReLU = class_dict_wrapper(nn.ReLU)
Tanh = class_dict_wrapper(nn.Tanh)
Sigmoid = class_dict_wrapper(nn.Sigmoid)
LeakyReLU = class_dict_wrapper(nn.LeakyReLU)
ELU = class_dict_wrapper(nn.ELU)
GELU = class_dict_wrapper(nn.GELU)
SiLU = class_dict_wrapper(nn.SiLU)

def get_activation(name: Optional[str]) -> Optional[Type[nn.Module]]:
    """
    Get a PyTorch activation function class from a string name.
    Args:
        name (Optional[str]): Name of the activation function (e.g., "relu", "
    "tanh", "sigmoid", etc.). If None or empty, returns None.


    Returns:
        Optional[Type[nn.Module]]: Corresponding PyTorch activation function class (not an
            instance) or None if not found. Callers should instantiate the returned class,
            e.g. ``get_activation("relu")()``.
    """
    if not name:
        return None
    name = name.lower()
    return {
        "relu": ReLU,
        "tanh": Tanh,
        "sigmoid": Sigmoid,
        "leaky_relu": LeakyReLU,
        "elu": ELU,
        "gelu": GELU,
        "silu": SiLU,
        None: None,
    }.get(name, None)

