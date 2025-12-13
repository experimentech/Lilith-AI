#!/usr/bin/env python3
"""
Test script for World Model Stage

Demonstrates:
- Entity extraction from text
- Spatial/temporal/causal relation extraction
- Pattern learning and retrieval
- Contrastive learning for world concepts
- Success score updates via reinforcement
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.world_model_stage import WorldModelStage
from lilith.embedding import PMFlowEmbeddingEncoder


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_basic_functionality():
    """Test basic world model functionality."""
    print_section("1. Initializing World Model Stage")
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(latent_dim=64, seed=42)
    
    # Create world model stage
    world_model = WorldModelStage(
        encoder=encoder,
        storage_path=Path("data/test_world_model"),
        use_sqlite=False,  # Use JSON for testing
        plasticity_enabled=True,
        enable_tracking=True,
    )
    
    stats = world_model.get_stats()
    print(f"✓ Stage initialized:")
    print(f"  - Latent dim: {stats['latent_dim']}")
    print(f"  - Seed patterns: {stats['total_patterns']}")
    print(f"  - Storage: {stats['storage']}")
    
    return world_model


def test_entity_extraction(world_model: WorldModelStage):
    """Test entity and relation extraction."""
    print_section("2. Entity and Relation Extraction")
    
    test_utterances = [
        "The book is on the table",
        "The cat is in the box under the bed",
        "Meet me after lunch at the cafe",
        "Rain causes flooding in the streets",
        "The door opened because of the wind",
    ]
    
    for utterance in test_utterances:
        print(f"Input: {utterance}")
        situation = world_model.process_utterance(utterance)
        
        print(f"  Entities: {len(situation.entities)}")
        for entity in situation.entities:
            print(f"    - {entity.name} ({entity.entity_type})")
        
        if situation.spatial_relations:
            print(f"  Spatial: {situation.spatial_relations[0].relation_type}")
        if situation.temporal_relations:
            print(f"  Temporal: {situation.temporal_relations[0].relation_type}")
        if situation.causal_relations:
            print(f"  Causal: {situation.causal_relations[0].relation_type}")
        
        print()


def test_pattern_learning(world_model: WorldModelStage):
    """Test learning new world situations."""
    print_section("3. Learning World Situations")
    
    # Learn some situations
    situations = [
        "The coffee cup is on the kitchen counter",
        "Keys are in the drawer next to the door",
        "Exercise before breakfast improves metabolism",
        "Study after dinner when it's quiet",
        "Sunshine causes plants to grow faster",
        "Lack of water prevents flower blooming",
    ]
    
    for desc in situations:
        situation = world_model.process_utterance(desc)
        pattern = world_model.learn_situation(situation, success_feedback=0.6)
        print(f"✓ Learned: {desc}")
        print(f"  Pattern ID: {pattern.pattern_id}")
        print(f"  Intent: {pattern.intent}")
        print(f"  Metadata: {pattern.metadata}")
        print()
    
    # Check stats
    stats = world_model.get_stats()
    print(f"Total patterns now: {stats['total_patterns']}")


def test_pattern_retrieval(world_model: WorldModelStage):
    """Test retrieving similar situations."""
    print_section("4. Retrieving Similar Situations")
    
    queries = [
        ("Where do I put my mug?", "spatial"),
        ("When should I workout?", "temporal"),
        ("Why won't my plants grow?", "causal"),
    ]
    
    for query, expected_type in queries:
        print(f"Query: {query}")
        print(f"Expected: {expected_type} patterns")
        
        results = world_model.retrieve_similar_situations(
            query_text=query,
            topk=3,
        )
        
        if results:
            print(f"  Found {len(results)} similar situations:")
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result.pattern.content[:60]}...")
                print(f"       Similarity: {result.similarity:.3f}, Confidence: {result.confidence:.3f}")
                print(f"       Intent: {result.pattern.intent}")
        else:
            print("  No similar situations found")
        
        print()


def test_success_updates(world_model: WorldModelStage):
    """Test reinforcement learning via success updates."""
    print_section("5. Reinforcement Learning")
    
    # Get a pattern
    results = world_model.retrieve_similar_situations(
        "Coffee on counter",
        topk=1,
    )
    
    if not results:
        print("⚠️  No patterns available for testing")
        return
    
    pattern = results[0].pattern
    print(f"Testing pattern: {pattern.content[:60]}")
    print(f"Initial success score: {pattern.success_score:.3f}")
    
    # Simulate positive feedback
    print("\nApplying positive feedback (+0.3)...")
    report = world_model.update_success(
        pattern_id=pattern.pattern_id,
        feedback=0.3,
        learning_rate=0.1,
        apply_plasticity=True,
    )
    
    # Get updated pattern
    results = world_model.retrieve_similar_situations(
        pattern.content,
        topk=1,
    )
    updated_pattern = results[0].pattern
    
    print(f"Updated success score: {updated_pattern.success_score:.3f}")
    print(f"Usage count: {updated_pattern.usage_count}")
    
    if report:
        print(f"\nPlasticity update applied:")
        print(f"  Type: {report.plasticity_type}")
        print(f"  Δ centers: {report.delta_centers:.6f}")
        print(f"  Δ mus: {report.delta_mus:.6f}")


def test_contrastive_learning(world_model: WorldModelStage):
    """Test contrastive learning for world concepts."""
    print_section("6. Contrastive Learning")
    
    print("Teaching the system that:")
    print("  - Spatial relations are similar to each other")
    print("  - Causal relations are similar to each other")
    print("  - Spatial relations are dissimilar from causal relations")
    
    # Define similar pairs (same relation type)
    similar_pairs = [
        ("The book is on the shelf", "The cup is on the table"),
        ("Rain causes flooding", "Sun causes heating"),
        ("Meet before noon", "Exercise after work"),
    ]
    
    # Define dissimilar pairs (different relation types)
    dissimilar_pairs = [
        ("The book is on the shelf", "Rain causes flooding"),
        ("Exercise after work", "The cup is on the table"),
        ("Sun causes heating", "Meet before noon"),
    ]
    
    print(f"\nApplying contrastive learning:")
    print(f"  Similar pairs: {len(similar_pairs)}")
    print(f"  Dissimilar pairs: {len(dissimilar_pairs)}")
    
    report = world_model.add_contrastive_pairs(
        similar_pairs=similar_pairs,
        dissimilar_pairs=dissimilar_pairs,
    )
    
    if report:
        print(f"\n✓ Contrastive update applied:")
        print(f"  Δ centers: {report.delta_centers:.6f}")
        print(f"  Δ mus: {report.delta_mus:.6f}")
        print(f"\nThis shapes the latent space so similar concepts cluster together!")
    else:
        print("\n⚠️  Contrastive learning not available (PMFlow 0.3.0+ required)")


def test_tracking(world_model: WorldModelStage):
    """Test entity and relation tracking across conversation."""
    print_section("7. Entity Tracking Across Conversation")
    
    # Clear previous tracking
    world_model.clear_tracking()
    
    conversation = [
        "There's a red car in the driveway",
        "The keys are on the kitchen counter",
        "I need to drive to the store after breakfast",
        "The store closes at 9pm",
    ]
    
    print("Simulating conversation:")
    for i, utterance in enumerate(conversation, 1):
        print(f"\n  Turn {i}: {utterance}")
        situation = world_model.process_utterance(utterance)
        
        # Show current context
        context = world_model.get_active_context()
        print(f"    Active entities: {context['num_active_entities']}")
        print(f"    Spatial relations: {context['num_spatial_relations']}")
        print(f"    Temporal relations: {context['num_temporal_relations']}")
        
        if context['entities']:
            print(f"    Entities tracked: {[e['name'] for e in context['entities']]}")
    
    # Show final context
    print("\nFinal world model state:")
    context = world_model.get_active_context()
    print(f"  Total entities: {context['num_active_entities']}")
    print(f"  Total relations: {context['num_spatial_relations'] + context['num_temporal_relations'] + context['num_causal_relations']}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  WORLD MODEL STAGE TEST SUITE")
    print("="*60)
    print("\nThis demonstrates Lilith's world grounding capability -")
    print("the missing piece that enables reasoning about concrete situations.")
    
    try:
        # Run tests
        world_model = test_basic_functionality()
        test_entity_extraction(world_model)
        test_pattern_learning(world_model)
        test_pattern_retrieval(world_model)
        test_success_updates(world_model)
        test_contrastive_learning(world_model)
        test_tracking(world_model)
        
        # Final stats
        print_section("Summary")
        stats = world_model.get_stats()
        print(f"Final stage statistics:")
        print(f"  Total patterns: {stats['total_patterns']}")
        print(f"  Average success: {stats['average_success']:.3f}")
        print(f"  Plasticity updates: {stats['plasticity_updates']}")
        print(f"  Latent dimension: {stats['latent_dim']}")
        
        print("\n✅ All tests completed successfully!")
        print("\nThe world model stage gives Lilith:")
        print("  - Spatial understanding (locations, containment)")
        print("  - Temporal reasoning (sequences, duration)")
        print("  - Causal understanding (cause-effect chains)")
        print("  - Entity tracking (objects, events, states)")
        print("\nThis fixes the 'aphantasia' - now symbols are grounded in world knowledge!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
