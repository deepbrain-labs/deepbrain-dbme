"""
This module will contain the logic for applying the different ablation settings.
"""

def apply_ablation(config, ablation_name):
    """
    Applies the ablation settings to the model configuration.
    """
    print(f"Applying ablation: {ablation_name}")
    if ablation_name == "no_consolidation":
        config["model"]["consolidation"] = {
            "enabled": False
        }
    elif ablation_name == "consolidator_variant_prototype":
        config["model"]["consolidation"] = {
            "enabled": True,
            "mode": "prototype"
        }
    elif "routing_variant" in ablation_name:
        variant = ablation_name.split("_")[-1]
        config["model"]["router"] = {
            "mode": variant
        }
    elif "forgetting_policy" in ablation_name:
        policy = ablation_name.split("forgetting_policy_")[-1]
        if "storage" not in config:
            config["storage"] = {}
        if "episodic_store" not in config["storage"]:
            config["storage"]["episodic_store"] = {}
        config["storage"]["episodic_store"]["eviction_policy"] = policy
    elif "compression_level" in ablation_name:
        level = int(ablation_name.split("_")[-1])
        config["model"]["hippocampal_encoder"] = {
            "slot_dim": level
        }
    elif "consolidation_frequency" in ablation_name:
        frequency = int(ablation_name.split("_")[-1])
        if "consolidation" not in config["model"]:
            config["model"]["consolidation"] = {}
        config["model"]["consolidation"]["frequency"] = frequency
    elif "insertion_mode" in ablation_name:
        mode = ablation_name.split("_")[-1]
        config["model"]["insertion_mode"] = mode
    elif "fusion_mode" in ablation_name:
        mode = ablation_name.split("fusion_mode_")[-1]
        config["model"]["language_model"] = {
            "fusion_mode": mode
        }
    elif ablation_name == "adversarial_stale_facts":
        config["evaluation"] = {
            "adversarial_stale_facts": True
        }
    elif "memory_budget" in ablation_name:
        budget = int(ablation_name.split("_")[-1])
        if "storage" not in config:
            config["storage"] = {}
        if "episodic_store" not in config["storage"]:
            config["storage"]["episodic_store"] = {}
        config["storage"]["episodic_store"]["capacity"] = budget
    return config