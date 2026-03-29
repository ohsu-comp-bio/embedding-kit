
import unittest
from embkit import factory

class TestFactory(unittest.TestCase):

    def test_ffnn_to_dict(self):

        l = factory.Linear(20, 1)
        data = l.to_dict()
        self.assertEqual(data["in_features"], 20)
        self.assertEqual(data["out_features"], 1)
        self.assertIn("__class__", data)

        cls = factory.Linear.from_dict(data)
        self.assertEqual(cls.in_features, 20)
        self.assertEqual(cls.out_features, 1)
    
    def test_modulelist_build(self):

        model = factory.build([
            factory.Linear(10, 20),
            factory.Linear(20, 1),
        ])
        self.assertEqual(len(model), 2)
        self.assertEqual(model[0].in_features, 10)
        self.assertEqual(model[0].out_features, 20)
        self.assertEqual(model[1].out_features, 1)
    

    def test_layer_build(self):

        ll = factory.LayerList([
            factory.Layer(10, op="linear", activation="relu")
        ])

        module = ll.build(100, 1)

        module_dict = module.to_dict()
        self.assertIn("args", module_dict)
        self.assertGreaterEqual(len(module_dict["args"]), 2)

        module2 = factory.build(module_dict)
        self.assertEqual(len(module2), len(module))