import torch
import sys
import os
import unittest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.hippocampal_encoder import HippocampalEncoder

class TestHippocampalEncoder(unittest.TestCase):
    def test_shapes_and_forward(self):
        input_dim = 128
        slot_dim = 256
        key_dim = 128
        batch_size = 4
        
        he = HippocampalEncoder(input_dim, slot_dim, key_dim, vae_mode=False)
        x = torch.randn(batch_size, input_dim)
        
        # Test forward
        key, slot, info = he(x)
        self.assertEqual(key.shape, (batch_size, key_dim))
        self.assertEqual(slot.shape, (batch_size, slot_dim))
        
        # Test write (single item)
        # write expects (input_dim) or (1, input_dim)
        x_single = torch.randn(input_dim)
        k, s, m = he.write(x_single, meta={'id': 1})
        self.assertEqual(k.shape, (1, key_dim))
        self.assertEqual(s.shape, (1, slot_dim))
        self.assertEqual(m['id'], 1)

    def test_vae_mode(self):
        input_dim = 128
        slot_dim = 256
        he = HippocampalEncoder(input_dim, slot_dim, vae_mode=True)
        x = torch.randn(2, input_dim)
        
        key, slot, info = he(x)
        self.assertTrue('mu' in info)
        self.assertTrue('logvar' in info)
        self.assertEqual(slot.shape, (2, slot_dim))

    def test_gradients(self):
        # Ensure gradients flow
        input_dim = 32
        he = HippocampalEncoder(input_dim, vae_mode=False)
        x = torch.randn(2, input_dim, requires_grad=True)
        
        key, slot, _ = he(x)
        loss = key.sum() + slot.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        # Check if weights have grad
        self.assertIsNotNone(he.trunk[0].weight.grad)

if __name__ == '__main__':
    unittest.main()