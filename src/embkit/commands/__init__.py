
"""
Command Line Methods

embkit matrix <> : Matrix methods for normalization and transformation

embkit model <> : Methods of model training and application

"""

from .model import model
from .matrix import matrix
from .cbio import cbio_cmd

__all__ = ["model", "matrix"]
