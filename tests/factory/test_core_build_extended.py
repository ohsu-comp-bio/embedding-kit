"""Extended tests for embkit.factory.core.build covering more branches."""

import unittest
from unittest import mock

from embkit.factory import nn_module
from embkit.factory import core
from torch import nn


@nn_module
class DummyObj(nn.Module):
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data

    @classmethod
    def from_dict(cls, d):
        return f"obj:{d['name']}"


class TestCoreBuildExtended(unittest.TestCase):
    def test_build_from_object_with_to_dict(self):
        # Object with .to_dict() should be handled like a dict
        obj = DummyObj({"__class__": DummyObj.__name__, "name": "obj"})
        result = core.build(obj)
        self.assertEqual(result, "obj:obj")

    def test_build_unknown_class_raises(self):
        desc = {"__class__": "Nonexistent", "foo": 1}
        with self.assertRaises(Exception) as ctx:
            core.build(desc)
        self.assertIn("Unknown layer type", str(ctx.exception))

    def test_build_unknown_str_returns_none(self):
        # get_activation returns None for unknown activation, so build should raise
        with self.assertRaises(Exception) as ctx:
            core.build("unknown_activation")
        self.assertIn("Invalid input for build function", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
