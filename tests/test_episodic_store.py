import unittest
import numpy as np
import os
import shutil
import tempfile
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.episodic_store import EpisodicStore

class TestEpisodicStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.test_dir, "test_log.jsonl")
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_insert_and_query(self):
        key_dim = 64
        slot_dim = 32
        store = EpisodicStore(key_dim, slot_dim, capacity=10, storage_path=self.log_path)
        
        # Insert 5 items
        for i in range(5):
            k = np.random.randn(key_dim).astype(np.float32)
            # Normalize for better IP search behavior usually, but not strictly required by FlatIP
            k = k / np.linalg.norm(k)
            s = np.random.randn(slot_dim).astype(np.float32)
            store.append(k, s, {'data': i})
            
        # Query
        q = np.random.randn(key_dim).astype(np.float32)
        results = store.query_by_key(q, top_k=2)
        self.assertTrue(len(results) <= 2)
        
    def test_eviction_fifo(self):
        key_dim = 16
        slot_dim = 16
        store = EpisodicStore(key_dim, slot_dim, capacity=5, storage_path=self.log_path, eviction_policy='fifo')
        
        # Insert 6 items, first one should be evicted
        ids = []
        for i in range(6):
            k = np.random.randn(key_dim).astype(np.float32)
            s = np.random.randn(slot_dim).astype(np.float32)
            ids.append(store.append(k, s, {'id_check': i}))
            
        self.assertEqual(len(store.store), 5)
        # Check that 0th added (id=ids[0]) is gone
        self.assertIsNone(store.get_entry(ids[0]))
        self.assertIsNotNone(store.get_entry(ids[5]))

    def test_eviction_importance(self):
        key_dim = 16
        slot_dim = 16
        store = EpisodicStore(key_dim, slot_dim, capacity=3, storage_path=self.log_path, eviction_policy='importance')
        
        ids = []
        # Add 3 items
        for i in range(3):
            k = np.random.randn(key_dim).astype(np.float32)
            s = np.random.randn(slot_dim).astype(np.float32)
            ids.append(store.append(k, s, {'i': i}))
            
        # Update meta 'ctr' for importance
        store.update_entry_meta(ids[0], {'ctr': 10}) # High importance
        store.update_entry_meta(ids[1], {'ctr': 0}) # Low importance
        store.update_entry_meta(ids[2], {'ctr': 5}) # Medium
        
        # Add 4th item
        k = np.random.randn(key_dim).astype(np.float32)
        s = np.random.randn(slot_dim).astype(np.float32)
        new_id = store.append(k, s)
        
        # Should have evicted ids[1] (ctr 0)
        self.assertIsNone(store.get_entry(ids[1]))
        self.assertIsNotNone(store.get_entry(ids[0])) # Kept
        self.assertIsNotNone(store.get_entry(ids[2])) # Kept

if __name__ == '__main__':
    unittest.main()
