
import unittest
from embkit import factory

class TestFactory(unittest.TestCase):

    def test_ffnn_to_dict(self):

        l = factory.Linear(20, 1)
        print(l.to_dict())

        cls = factory.Linear.from_dict(l.to_dict())
        print(cls)
    
    def test_modulelist_build(self):

        model = factory.build([
            factory.Linear(10, 20),
            factory.Linear(20, 1),
        ])
        print(model)
    

    def test_layer_build(self):

        ll = factory.LayerList([
            factory.Layer(10, op="linear", activation="relu")
        ])

        module = ll.build(100, 1)

        module_dict = module.to_dict()
        print(module_dict)

        module2 = factory.build(module_dict)
        print(module2)