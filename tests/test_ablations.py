import unittest
import yaml
from src.ablations import apply_ablation

class TestAblations(unittest.TestCase):

    def setUp(self):
        with open('configs/base_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

    def test_no_consolidation_ablation(self):
        config = apply_ablation(self.config.copy(), "no_consolidation")
        self.assertFalse(config["model"]["consolidation"]["enabled"])


    def test_routing_variant_ablation(self):
        config = apply_ablation(self.config.copy(), "routing_variant_heuristic")
        self.assertEqual(config["model"]["router"]["mode"], "heuristic")

    def test_forgetting_policy_ablation(self):
        config = apply_ablation(self.config.copy(), "forgetting_policy_time_decay")
        self.assertEqual(config["storage"]["episodic_store"]["eviction_policy"], "time_decay")

    def test_compression_level_ablation(self):
        config = apply_ablation(self.config.copy(), "compression_level_128")
        self.assertEqual(config["model"]["hippocampal_encoder"]["slot_dim"], 128)

    def test_consolidation_frequency_ablation(self):
        config = apply_ablation(self.config.copy(), "consolidation_frequency_50")
        self.assertEqual(config["model"]["consolidation"]["frequency"], 50)

    def test_insertion_mode_ablation(self):
        config = apply_ablation(self.config.copy(), "insertion_mode_per-token")
        self.assertEqual(config["model"]["insertion_mode"], "per-token")

    def test_fusion_mode_ablation(self):
        config = apply_ablation(self.config.copy(), "fusion_mode_token_injection")
        self.assertEqual(config["model"]["language_model"]["fusion_mode"], "token_injection")

    def test_adversarial_stale_facts_ablation(self):
        config = apply_ablation(self.config.copy(), "adversarial_stale_facts")
        self.assertTrue(config["evaluation"]["adversarial_stale_facts"])

    def test_memory_budget_ablation(self):
        config = apply_ablation(self.config.copy(), "memory_budget_5000")
        self.assertEqual(config["storage"]["episodic_store"]["capacity"], 5000)

if __name__ == '__main__':
    unittest.main()
