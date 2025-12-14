import unittest
import numpy as np
import torch
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.consolidator import Consolidator as PrototypeConsolidator

class TestConsolidator(unittest.TestCase):
    def test_prototype_consolidator(self):
        # Needs sklearn
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("Skipping PrototypeConsolidator test: sklearn not found")
            return

        slot_dim = 16
        key_dim = 10
        n_proto = 5
        pc = PrototypeConsolidator(mode='prototype', n_prototypes=n_proto, dimension=slot_dim)
        
        # 20 samples
        keys = torch.randn(20, key_dim)
        slots = torch.randn(20, slot_dim)
        
        prototypes, labels = pc.find_prototypes(keys, slots)
        
        self.assertIsInstance(prototypes, list)
        self.assertEqual(len(prototypes), n_proto)
        
        # Check the shape of the returned keys and slots
        proto_key, proto_slot = prototypes[0]
        self.assertEqual(proto_key.shape, (key_dim,))
        self.assertEqual(proto_slot.shape, (slot_dim,))

        # Check the labels
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.shape, (20,))

if __name__ == '__main__':
    unittest.main()
