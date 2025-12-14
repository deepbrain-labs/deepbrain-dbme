import unittest
import numpy as np
import os
import shutil
import tempfile
import sys
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.episodic_store import EpisodicStore

class TestEpisodicStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_insert_and_query(self):
        key_dim = 64
        slot_dim = 32
        store = EpisodicStore(key_dim, slot_dim, capacity=10)
        
        for i in range(5):
            k = torch.randn(key_dim)
            s = torch.randn(slot_dim)
            store.add(k, s, {'data': i})
            
        q = torch.randn(key_dim)
        results = store.retrieve(q, k=2)
        self.assertTrue(len(results["slots"]) <= 2)
        
    def test_eviction_fifo(self):
        key_dim = 16
        slot_dim = 16
        store = EpisodicStore(key_dim, slot_dim, capacity=5, eviction_policy='fifo')
        
        ids = []
        for i in range(6):
            k = torch.randn(key_dim)
            s = torch.randn(slot_dim)
            ids.append(store.add(k, s, {'id_check': i}))
            
        self.assertEqual(store.size, 5)
        self.assertIsNone(store.get_entry(ids[0]))
        self.assertIsNotNone(store.get_entry(ids[5]))

    def test_eviction_importance(self):
        key_dim = 16
        slot_dim = 16
        store = EpisodicStore(key_dim, slot_dim, capacity=3, eviction_policy='importance')
        
        ids = []
        for i in range(3):
            k = torch.randn(key_dim)
            s = torch.randn(slot_dim)
            ids.append(store.add(k, s, {'i': i, 'importance': i}))
            
        k = torch.randn(key_dim)
        s = torch.randn(slot_dim)
        store.add(k, s, {'i': 3, 'importance': 3})
        
        self.assertEqual(store.size, 3)
        self.assertIsNone(store.get_entry(ids[0]))
        self.assertIsNotNone(store.get_entry(ids[1]))
        self.assertIsNotNone(store.get_entry(ids[2]))

if __name__ == '__main__':
    unittest.main()
