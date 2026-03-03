**Factory Decorators**

This file contains examples and usage notes for the `factory` decorators.

**Overview**

- **`nn_module`**: A class decorator that registers a class in the global factory registry under its module-qualified name (module.ClassName). It also wraps the class's `to_dict` output to include a `"__class__"` key.
- **`nn_module_named`**: A decorator class that accepts an explicit name and registers the class under that name instead of the module-qualified name.

Both decorators require the decorated class to implement `to_dict(self) -> dict` and `@classmethod from_dict(cls, data: dict) -> object`. If those methods are missing the decorator will raise `ValueError` at decoration time.

**Importing the decorators**

Use the public factory package exports:

```python
from embkit import factory
```

This will give access to `factory.nn_module` and `factory.nn_module_named`

**Simple Example (automatic name)**

```python
from embkit import factory
import torch.nn as nn

@factory.nn_module
class MyLayer(nn.Module):
	def __init__(self, hidden=32):
		super().__init__()
		self.hidden = hidden

	def forward(self, x):
		# Minimal forward that demonstrates a shape-changing op.
		# In real classes you would use actual nn.Modules (Linear/Conv/etc.).
		return x

	def to_dict(self):
		# Return a serializable representation. The decorator will add
		# the top-level "__class__" key automatically.
		return {"params": {"hidden": self.hidden}}

	@classmethod
	def from_dict(cls, data):
		params = data.get("params", {})
		return cls(**params)

inst = MyLayer(64)

factory.save(inst, "my_module.model")

# Reconstruct using the factory registry

new_inst = factory.load("my_module.model")

```

When using `@factory.nn_module` the registry key will be the module-qualified name, e.g. `embkit.examples.factory.MyLayer` (see `embkit.factory.get_class_name` for the exact naming rule). This may not always be avalible (example Jupyter notebooks), so it is possible to manually assert the classes name in 
registery with `@factory.nn_module_named`

**Explicit name example (`nn_module_named`)**

```python
from embkit import factory
import torch.nn as nn

@factory.nn_module_named("custom.layers.ConvX")
class ConvX(nn.Module):
	def __init__(self, channels=16):
		super().__init__()
		self.channels = channels

	def forward(self, x):
		# Placeholder forward; replace with real conv ops in production.
		return x

	def to_dict(self):
		return {"params": {"channels": self.channels}}

	@classmethod
	def from_dict(cls, data):
		return cls(**data.get("params", {}))


```

**Notes & Best Practices**

- **Requirement**: Decorated classes must implement `to_dict` and `from_dict`.
- **Serialization shape**: The decorator expects `to_dict` to return a mapping; it will add `"__class__"` to that mapping. Keep your payloads stable so `from_dict` can read them reliably.
- **Explicit naming**: Use `nn_module_named` when you want stable registry keys that are independent of Python module layout. This is useful for classes written in Jupyter notebooks

