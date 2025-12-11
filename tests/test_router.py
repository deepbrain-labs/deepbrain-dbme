import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.router import Router

class TestRouter(unittest.TestCase):
    def test_routing(self):
        dim = 32
        router = Router(dim)
        x = torch.randn(5, dim)
        
        choices, probs = router.route(x)
        self.assertEqual(choices.shape, (5,))
        self.assertEqual(probs.shape, (5, 2))
        
        # Check sum to 1
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(5)))

    def test_gradients(self):
        dim = 32
        router = Router(dim)
        x = torch.randn(2, dim, requires_grad=True)
        
        choices, probs = router.route(x)
        loss = probs.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)

if __name__ == '__main__':
    unittest.main()
