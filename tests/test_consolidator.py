import unittest
import numpy as np
import torch
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.consolidator import PrototypeConsolidator, DistillationConsolidator

class TestConsolidator(unittest.TestCase):
    def test_prototype_consolidator(self):
        # Needs sklearn
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("Skipping PrototypeConsolidator test: sklearn not found")
            return

        dim = 16
        n_proto = 5
        pc = PrototypeConsolidator(n_prototypes=n_proto, dimension=dim)
        
        # 20 samples
        slots = np.random.randn(20, dim).astype(np.float32)
        
        pc.consolidate(slots)
        
        protos = pc.get_kstore_reps()
        self.assertEqual(protos.shape, (n_proto, dim))
        
        # Test query
        q = np.random.randn(dim).astype(np.float32)
        dists, indices = pc.query_kstore(q, top_k=2)
        self.assertEqual(len(indices), 2)

    def test_distillation_consolidator(self):
        key_dim = 16
        slot_dim = 32
        dc = DistillationConsolidator(key_dim, slot_dim)
        
        # Mock data
        keys = np.random.randn(20, key_dim).astype(np.float32)
        slots = np.random.randn(20, slot_dim).astype(np.float32)
        
        # Train
        dc.consolidate(keys, slots, epochs=2, batch_size=5)
        
        # Query
        q_tensor = torch.randn(1, key_dim)
        pred_slot = dc.query_kstore(q_tensor)
        self.assertEqual(pred_slot.shape, (1, slot_dim))

if __name__ == '__main__':
    unittest.main()
