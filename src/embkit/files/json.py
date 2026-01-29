
import json
import torch

class StringTrimEncoder(json.JSONEncoder):
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