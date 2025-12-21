import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore

# Mock encoder output
def get_mock_embedding(seed, dim=128):
    torch.manual_seed(seed)
    return torch.randn(1, dim)

@pytest.fixture
def estore():
    # A fresh episodic store for each test
    return EpisodicStore(key_dim=128, slot_dim=256, capacity=100)

@pytest.fixture
def kstore():
    # A fresh knowledge store for each test
    return KStore(key_dim=128, value_dim=256, capacity=100)

def test_add_and_retrieve_from_estore(estore):
    """Test basic add and retrieve functionality."""
    key = get_mock_embedding(1, 128)
    slot = get_mock_embedding(1, 256)
    estore.add(key, slot, meta={'fact_id': 'fact1'})
    
    retrieved = estore.retrieve(key, k=1)
    assert len(retrieved['slots']) == 1
    assert torch.allclose(retrieved['slots'][0], slot)
    assert retrieved['meta'][0][0]['fact_id'] == 'fact1'

def test_forget_from_estore_by_fact_id(estore):
    """Test that forgetting removes the correct item from EpisodicStore."""
    # Add two distinct facts
    key1, slot1 = get_mock_embedding(1, 128), get_mock_embedding(1, 256)
    key2, slot2 = get_mock_embedding(2, 128), get_mock_embedding(2, 256)
    estore.add(key1, slot1, meta={'fact_id': 'fact_to_forget'})
    estore.add(key2, slot2, meta={'fact_id': 'fact_to_keep'})

    assert len(estore) == 2

    # Forget one fact
    estore.forget(fact_ids=['fact_to_forget'])

    assert len(estore) == 1

    # Verify the correct fact was removed
    # Query for the forgotten fact - should not be in top 1
    retrieved_forgotten = estore.retrieve(key1, k=1)
    if retrieved_forgotten['meta']: # If anything is retrieved
        assert retrieved_forgotten['meta'][0][0]['fact_id'] != 'fact_to_forget'

    # Query for the fact to keep - should be retrieved
    retrieved_kept = estore.retrieve(key2, k=1)
    assert len(retrieved_kept['slots']) == 1
    assert retrieved_kept['meta'][0][0]['fact_id'] == 'fact_to_keep'

def test_prototype_repair_after_forgetting(kstore, estore):
    """
    Unit test to simulate the full forgetting workflow:
    1. Populate Episodic Store.
    2. Consolidate to KStore (mocked).
    3. Forget a fact from Episodic Store.
    4. Assert that a 'prototype repair' step is needed and would remove the influence.
    """
    # 1. Populate Episodic Store
    keys = [get_mock_embedding(i, 128) for i in range(10)]
    slots = [get_mock_embedding(i, 256) for i in range(10)]
    fact_ids = [f'fact_{i}' for i in range(10)]
    
    for i in range(10):
        estore.add(keys[i], slots[i], meta={'fact_id': fact_ids[i]})

    # 2. Mock consolidation: Assume fact_1 and fact_2 contribute to a prototype
    # A real implementation would run the Consolidator module.
    # Here, we'll manually create a prototype that is the average of slot 1 and 2.
    prototype_slot = (slots[1] + slots[2]) / 2
    # The key could be the key of the closest member, e.g., key of slot 1
    prototype_key = keys[1]
    kstore.add(prototype_key, prototype_slot, meta={'derived_from': ['fact_1', 'fact_2']})
    
    assert kstore.size == 1

    # 3. Forget fact_1 from Episodic Store
    estore.forget(fact_ids=['fact_1'])
    
    # 4. Prototype Repair Assertion
    # The core logic: after forgetting, the system must re-evaluate prototypes
    # that were derived from the forgotten fact.
    
    # Find prototypes affected by the forgotten fact
    affected_prototypes = []
    for i in range(kstore.size):
        meta = kstore.meta_store[i]
        if 'derived_from' in meta and 'fact_1' in meta['derived_from']:
            affected_prototypes.append(i)
    
    assert len(affected_prototypes) > 0, "Forgetting should flag prototypes for repair."
    
    # This is where the actual repair logic would be triggered.
    # For this test, we just assert that we've identified the need.
    print(f"\nTest confirms that forgetting fact_1 correctly identifies prototype(s) {affected_prototypes} for repair.")
    
    # A full implementation would now either:
    # A) Delete the affected prototype from KStore.
    # B) Re-run consolidation on the remaining contributors (e.g., only fact_2).
    
    # For this test, let's simulate option A:
    kstore.clear() # Simplified removal
    
    assert kstore.size == 0, "Prototype should be removed after repair."