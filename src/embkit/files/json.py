
"""
JSON helpers.

Utilities for serializing objects that aren't natively JSON-serializable —
notably PyTorch tensors — into plain, human-readable JSON with trimmed floats.
"""
import json
import torch

class StringTrimEncoder(json.JSONEncoder):
    """
    A JSON encoder that serializes PyTorch tensors and trims float precision.

    Standard ``json`` can't handle tensors, so this encoder converts them to
    lists recursively and rounds every float to a fixed number of decimal
    places (``trim``, default 5). Handy for writing model descriptions and
    diagnostics out as clean, portable JSON files.
    """
    def __init__(self, *args, trim=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim = trim

    def default(self, obj):
        # Handle PyTorch tensors
        if isinstance(obj, torch.Tensor):
            return self.encode_list(obj.tolist())
        return super().default(obj)

    def encode_list(self, data):
        if isinstance(data, list):
            return [self.encode_list(i) for i in data]
        # Trim float to X decimal places
        return float(f"{data:.{self.trim}f}")