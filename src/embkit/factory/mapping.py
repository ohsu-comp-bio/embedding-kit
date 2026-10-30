
"""
Docstring for embkit.factory.mapping

Lightweight mappings for nn module, designed to 
provide a consistent way to do to_dict and from_dict methods for nn modules.
"""

from typing import Optional, Type

from torch import nn

from .registry import nn_module, get_class_name, class_dict_wrapper, CLASS_REGISTRY

def clean_params(params):
    """
    Strip the registry's ``"__class__"`` key from a serialized parameter dict.

    The :func:`~embkit.factory.registry.nn_module` decorator tags every
    ``to_dict`` output with a ``"__class__"`` identifier so that
    :func:`~embkit.factory.core.build` can resolve it back to the correct class.
    When a class rebuilds itself from that dict (via ``from_dict``), the
    ``"__class__"`` key is not a real constructor argument — this helper removes
    it so the remaining entries can be unpacked straight into ``cls(**params)``.

    Parameters:
        params: A serialized parameter dict, optionally including ``"__class__"``.

    Returns:
        A new dict identical to ``params`` but without the ``"__class__"`` key.
    """
    out = {}
    for k, v in params.items():
        if k not in ["__class__"]:
            out[k] = v
    return out

@nn_module
class Linear(nn.Linear):
    """
    A serializable ``nn.Linear`` layer.

    This is the workhorse connection layer of embkit's factory. It subclasses
    ``nn.Linear`` only to add the ``to_dict``/``from_dict`` pair required by the
    registry, so that a layer's configuration (dimensions, bias flag) survives a
    round-trip through ``save``/``load``. Use this rather than a raw
    ``nn.Linear`` whenever the layer needs to be part of a (de)serializable
    model description.

    Note:
        The serialized params (``in_features``, ``out_features``, ``bias``) are
        what gets rebuilt; learned weights are restored separately from the state
        dict during :func:`~embkit.factory.core.load`.
    """

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
    """
    A serializable ``nn.BatchNorm1d`` layer.

    Batch normalization stabilizes training by standardizing each feature's
    activations across a batch. This subclass mirrors
    :class:`Linear` — it adds ``to_dict``/``from_dict`` so the layer can be part
    of a model description that is saved and rebuilt, while retaining its running
    statistics (which are weights and are restored from the state dict).
    """

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
    """
    A serializable ``nn.Sequential`` container.

    A :class:`Sequential` stacks a series of layers into a single ordered
    pipeline — the way an embkit model's layers are composed. This subclass adds
    the ``to_dict``/``from_dict`` pair so that an entire stack can be captured in
    a model description and reconstructed element-by-element: each child module is
    resolved through the :data:`~embkit.factory.registry.CLASS_REGISTRY` rather
    than relying on pickled classes.

    Note:
        Reconstruction relies on every child module itself being registry-
        serializable (e.g. :class:`Linear`, :class:`BatchNorm1d`, or one of the
        activation wrappers below).
    """

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

