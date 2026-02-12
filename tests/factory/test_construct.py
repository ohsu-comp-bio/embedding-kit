
import unittest
from embkit import factory

class TestH5(unittest.TestCase):

    def test_ffnn_to_dict(self):

        l = factory.Linear(20, 1)
        print(l.to_dict())

        cls = factory.Linear.from_dict(l.to_dict())
        print(cls)
    
    def test_layer_build(self):

        model = factory.build([
            factory.Linear(10, 20),
            factory.Linear(20, 1),
        ])
        print(model)