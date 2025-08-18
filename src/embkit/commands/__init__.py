
"""
Command Line Methods

embkit matrix <> : Matrix methods for normalization and transformation

embkit model <> : Methods of model training and application

"""

from .model import model
from .matrix import matrix

__all__ = ["model", "matrix"]
