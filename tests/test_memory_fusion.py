import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.memory_fusion import TokenFusion, AdapterFusion

class TestMemoryFusion(unittest.TestCase):
    def test_token_fusion(self):
        tf = TokenFusion()
        # B=2, L=10, D=32
        lm_emb = torch.randn(2, 10, 32)
        # B=2, K=5, D=32 (5 memories)
        mem_emb = torch.randn(2, 5, 32)
        
        fused = tf(lm_emb, mem_emb)
        self.assertEqual(fused.shape, (2, 15, 32))

    def test_adapter_fusion(self):
        dim = 32
        af = AdapterFusion(embed_dim=dim, slot_dim=dim, num_heads=2)
        
        lm_states = torch.randn(2, 10, dim)
        mem_slots = torch.randn(2, 5, dim)
        
        fused = af(lm_states, mem_slots)
        self.assertEqual(fused.shape, (2, 10, dim))
        
        # Test gating init
        # alpha init 0 -> output should be exactly input (if residual structure matches)
        # Actually Pytorch MHA might have init weights that produce non-zero output, 
        # but alpha multiplication makes it 0.
        self.assertTrue(torch.allclose(fused, lm_states))
        
        # Change alpha to non-zero
        with torch.no_grad():
            af.alpha.fill_(1.0)
            
        fused_2 = af(lm_states, mem_slots)
        self.assertFalse(torch.allclose(fused_2, lm_states))

    def test_gradients(self):
        dim = 16
        af = AdapterFusion(embed_dim=dim, slot_dim=dim)
        lm = torch.randn(2, 5, dim, requires_grad=True)
        mem = torch.randn(2, 3, dim, requires_grad=True)
        
        # set alpha non-zero
        with torch.no_grad():
            af.alpha.fill_(0.5)
            
        out = af(lm, mem)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(lm.grad)
        self.assertIsNotNone(mem.grad)
        self.assertIsNotNone(af.alpha.grad)

if __name__ == '__main__':
    unittest.main()
