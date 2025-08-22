
"""
Command Line Methods

embkit matrix <> : Matrix methods for normalization and transformation

embkit model <> : Methods of model training and application

embkit cbio <> : cBIO methods for data querying and download

"""

from .model import model
from .matrix import matrix
from .cbio import cbio_cmd

__all__ = ["model", "matrix"]
