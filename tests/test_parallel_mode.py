"""
Test Automatic Parallel Mode

Validates that ResponseComposer automatically uses parallel mode
when composition_mode="parallel" and concept_store is provided.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.production_concept_store import ProductionConceptStore
from lilith.response_composer import ResponseComposer
from lilith.conversation_state import ConversationState
from lilith.response_fragments import ResponseFragmentStore


def test_automatic_parallel_mode():
    """Test that parallel mode is automatically invoked"""
    print("\n" + "="*60)
    print("TEST: Automatic Parallel Mode")
    print("="*60)
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_concepts.db"
    
    try:
        # Create concept store
        concept_store = ProductionConceptStore(
            encoder,
            str(db_path),
            consolidation_threshold=0.85
        )
        
        # Add a test concept
        concept_store.add_concept(
            "machine learning",
            ["learns from data", "branch of AI"],
            source="test"
        )
        
        # Create fragment store
        fragment_store = ResponseFragmentStore(encoder)
        
        # Create conversation state
        state = ConversationState(encoder)
        
        # Create composer with PARALLEL mode
        composer = ResponseComposer(
            fragment_store=fragment_store,
            conversation_state=state,
            composition_mode="parallel",  # Enable parallel mode
            concept_store=concept_store,
            enable_compositional=True,
            semantic_encoder=encoder
        )
        
        print("\n✅ Composer created with parallel mode")
        print(f"   composition_mode: {composer.composition_mode}")
        print(f"   concept_store: {composer.concept_store is not None}")
        print(f"   template_composer: {composer.template_composer is not None}")
        
        # Generate a response
        print("\n🧪 Testing automatic parallel invocation...")
        response = composer.compose_response(
            context="User asked about ML",
            user_input="what is machine learning"
        )
        
        print(f"\n📊 Response generated:")
        print(f"   Text: {response.text}")
        print(f"   Is fallback: {response.is_fallback}")
        print(f"   Confidence: {response.confidence:.3f}")
        
        # Check metrics
        metrics = composer.get_metrics()
        print(f"\n📈 Metrics after 1 response:")
        print(f"   Parallel uses: {metrics['parallel_uses']}")
        print(f"   Pattern count: {metrics['pattern_count']}")
        print(f"   Concept count: {metrics['concept_count']}")
        
        # Verify parallel mode was used
        if metrics['parallel_uses'] == 1:
            print("\n✅ Parallel mode automatically invoked!")
        else:
            print(f"\n❌ Parallel mode NOT invoked (parallel_uses={metrics['parallel_uses']})")
            return False
        
        # Verify one of the approaches was used
        if metrics['pattern_count'] + metrics['concept_count'] == 1:
            print("✅ One approach selected from parallel execution")
            if metrics['concept_count'] == 1:
                print("   → Concept-based approach used")
            else:
                print("   → Pattern-based approach used")
        else:
            print(f"❌ Unexpected counts: pattern={metrics['pattern_count']}, concept={metrics['concept_count']}")
            return False
        
        concept_store.close()
        
        print("\n" + "="*60)
        print("✅ AUTOMATIC PARALLEL MODE TEST PASSED")
        print("="*60)
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_non_parallel_mode():
    """Test that non-parallel modes still work"""
    print("\n" + "="*60)
    print("TEST: Non-Parallel Mode (Traditional)")
    print("="*60)
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_concepts.db"
    
    try:
        # Create concept store
        concept_store = ProductionConceptStore(
            encoder,
            str(db_path),
            consolidation_threshold=0.85
        )
        
        # Add a test concept
        concept_store.add_concept(
            "neural networks",
            ["inspired by brain", "uses layers"],
            source="test"
        )
        
        # Create fragment store
        fragment_store = ResponseFragmentStore(encoder)
        
        # Create conversation state
        state = ConversationState(encoder)
        
        # Create composer with ADAPTIVE mode (not parallel)
        composer = ResponseComposer(
            fragment_store=fragment_store,
            conversation_state=state,
            composition_mode="adaptive",  # NOT parallel
            concept_store=concept_store,
            enable_compositional=True,
            semantic_encoder=encoder
        )
        
        print("\n✅ Composer created with adaptive mode")
        print(f"   composition_mode: {composer.composition_mode}")
        
        # Generate a response
        print("\n🧪 Testing traditional (non-parallel) execution...")
        response = composer.compose_response(
            context="User asked about neural networks",
            user_input="what are neural networks"
        )
        
        print(f"\n📊 Response generated:")
        print(f"   Text: {response.text}")
        print(f"   Is fallback: {response.is_fallback}")
        
        # Check metrics
        metrics = composer.get_metrics()
        print(f"\n📈 Metrics after 1 response:")
        print(f"   Parallel uses: {metrics['parallel_uses']}")
        
        # Verify parallel mode was NOT used
        if metrics['parallel_uses'] == 0:
            print("\n✅ Parallel mode correctly NOT invoked for adaptive mode")
        else:
            print(f"\n❌ Parallel mode incorrectly invoked (parallel_uses={metrics['parallel_uses']})")
            return False
        
        concept_store.close()
        
        print("\n" + "="*60)
        print("✅ NON-PARALLEL MODE TEST PASSED")
        print("="*60)
        return True
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PARALLEL MODE INTEGRATION TESTS")
    print("="*60)
    
    try:
        result1 = test_automatic_parallel_mode()
        result2 = test_non_parallel_mode()
        
        if result1 and result2:
            print("\n" + "="*60)
            print("✅ ALL PARALLEL MODE TESTS PASSED")
            print("="*60)
            print("\nParallel mode integration complete:")
            print("  ✓ Automatic invocation when mode='parallel'")
            print("  ✓ Metrics tracking working")
            print("  ✓ Non-parallel modes unaffected")
        else:
            print("\n" + "="*60)
            print("❌ SOME TESTS FAILED")
            print("="*60)
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
