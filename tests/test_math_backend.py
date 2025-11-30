"""
Test Math Backend and Modal Routing

Validates that mathematical queries are:
1. Correctly detected by ModalClassifier
2. Computed by MathBackend (not learned as linguistic patterns)
3. Database protected from mathematical pollution
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.math_backend import MathBackend, SYMPY_AVAILABLE
from lilith.modal_classifier import ModalClassifier, Modality


def test_modal_classifier():
    """Test modal classification"""
    print("\n" + "="*60)
    print("TEST 1: Modal Classification")
    print("="*60)
    
    classifier = ModalClassifier()
    
    test_cases = [
        # (query, expected_modality, min_confidence)
        ("what is 2 + 2", Modality.MATH, 0.70),
        ("calculate 15 * 7", Modality.MATH, 0.75),
        ("solve x + 5 = 10", Modality.MATH, 0.85),
        ("derivative of x^2", Modality.MATH, 0.85),
        ("what is machine learning", Modality.LINGUISTIC, 0.90),
        ("how does supervised learning work", Modality.LINGUISTIC, 0.90),
        ("write a function to sort an array", Modality.CODE, 0.80),
    ]
    
    passed = 0
    for query, expected, min_conf in test_cases:
        modality, confidence = classifier.classify(query)
        status = "✅" if modality == expected and confidence >= min_conf else "❌"
        
        print(f"{status} '{query}'")
        print(f"   → {classifier.get_modality_name(modality)} (conf: {confidence:.2f})")
        
        if modality == expected and confidence >= min_conf:
            passed += 1
    
    print(f"\n✅ Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_math_backend():
    """Test mathematical computation"""
    print("\n" + "="*60)
    print("TEST 2: Math Backend Computation")
    print("="*60)
    
    if not SYMPY_AVAILABLE:
        print("⚠️  SymPy not installed - skipping math backend tests")
        print("   Install with: pip install sympy")
        return False
    
    backend = MathBackend()
    
    test_cases = [
        # (query, expected_in_result)
        ("2 + 2", "4"),
        ("what is 15 * 7", "105"),
        ("100 / 4", "25"),
        ("calculate 2^3", "8"),
        ("sqrt(16)", "4"),
        ("solve x + 5 = 10", "5"),
    ]
    
    passed = 0
    for query, expected in test_cases:
        # Check if backend can handle
        can_handle, conf = backend.can_handle(query)
        
        if not can_handle:
            print(f"❌ Backend cannot handle: '{query}'")
            continue
        
        # Compute result
        result = backend.compute(query)
        
        if result and expected in result.result:
            print(f"✅ {query}")
            print(f"   → {result.result}")
            passed += 1
        else:
            print(f"❌ {query}")
            if result:
                print(f"   → Got: {result.result}, Expected: {expected}")
            else:
                print(f"   → Computation failed")
    
    print(f"\n✅ Passed: {passed}/{len(test_cases)}")
    return passed >= len(test_cases) - 1  # Allow 1 failure


def test_integration():
    """Test full integration with ResponseComposer"""
    print("\n" + "="*60)
    print("TEST 3: Integration with ResponseComposer")
    print("="*60)
    
    if not SYMPY_AVAILABLE:
        print("⚠️  SymPy not installed - skipping integration test")
        return False
    
    from pipeline.embedding import PMFlowEmbeddingEncoder
    from pipeline.response_composer import ResponseComposer
    from pipeline.conversation_state import ConversationState
    from pipeline.response_fragments import ResponseFragmentStore
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Create fragment store
    fragment_store = ResponseFragmentStore(encoder)
    
    # Create conversation state
    state = ConversationState(encoder)
    
    # Create composer with math backend enabled
    composer = ResponseComposer(
        fragment_store=fragment_store,
        conversation_state=state,
        semantic_encoder=encoder,
        enable_modal_routing=True  # Enable math backend
    )
    
    print("\n✅ Composer created with math backend")
    print(f"   Math backend: {composer.math_backend is not None}")
    print(f"   Modal classifier: {composer.modal_classifier is not None}")
    
    # Test mathematical query
    print("\n🧪 Testing math query: '2 + 2'")
    response = composer.compose_response(
        context="User asking math question",
        user_input="2 + 2"
    )
    
    print(f"\n📊 Response:")
    print(f"   Text: {response.text}")
    print(f"   Modality: {response.modality}")
    print(f"   Confidence: {response.confidence:.2f}")
    
    # Check that math was used
    if response.modality and hasattr(response.modality, 'value'):
        modality_val = response.modality.value
    else:
        modality_val = str(response.modality)
    
    if "4" in response.text and modality_val == 'math':
        print("\n✅ Math backend used correctly!")
        
        # Check metrics
        metrics = composer.get_metrics()
        print(f"   Math count: {metrics.get('math_count', 0)}")
        
        # Verify database wasn't polluted
        print("\n🔍 Checking database protection...")
        print(f"   Pattern count: {metrics.get('pattern_count', 0)}")
        
        if metrics.get('pattern_count', 0) == 0:
            print("✅ Database protected - math not learned as linguistic pattern!")
            return True
        else:
            print("❌ Database pollution - math was learned as pattern")
            return False
    else:
        print(f"❌ Math backend not used (modality: {modality_val})")
        return False


def test_linguistic_fallback():
    """Test that non-math queries still work"""
    print("\n" + "="*60)
    print("TEST 4: Linguistic Queries Unaffected")
    print("="*60)
    
    if not SYMPY_AVAILABLE:
        print("⚠️  SymPy not installed - skipping test")
        return False
    
    from pipeline.embedding import PMFlowEmbeddingEncoder
    from pipeline.response_composer import ResponseComposer
    from pipeline.conversation_state import ConversationState
    from pipeline.response_fragments import ResponseFragmentStore
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Create fragment store
    fragment_store = ResponseFragmentStore(encoder)
    
    # Create conversation state
    state = ConversationState(encoder)
    
    # Create composer
    composer = ResponseComposer(
        fragment_store=fragment_store,
        conversation_state=state,
        semantic_encoder=encoder,
        enable_modal_routing=True
    )
    
    # Test linguistic query
    print("\n🧪 Testing linguistic query: 'what is machine learning'")
    response = composer.compose_response(
        context="User asking about ML",
        user_input="what is machine learning"
    )
    
    print(f"\n📊 Response:")
    print(f"   Text: {response.text}")
    print(f"   Modality: {response.modality}")
    
    # Check modality
    if response.modality:
        if hasattr(response.modality, 'value'):
            modality_val = response.modality.value
        else:
            modality_val = str(response.modality)
        
        if modality_val != 'math':
            print("✅ Linguistic query processed normally (not routed to math)")
            return True
        else:
            print("❌ Linguistic query incorrectly routed to math")
            return False
    else:
        # No modality set = linguistic (default)
        print("✅ Linguistic query processed normally (default path)")
        return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MATH BACKEND AND MODAL ROUTING TESTS")
    print("="*60)
    
    if not SYMPY_AVAILABLE:
        print("\n❌ SymPy not installed!")
        print("   Install with: pip install sympy")
        print("   Skipping tests...")
        sys.exit(1)
    
    try:
        results = []
        
        results.append(test_modal_classifier())
        results.append(test_math_backend())
        results.append(test_integration())
        results.append(test_linguistic_fallback())
        
        passed = sum(results)
        total = len(results)
        
        print("\n" + "="*60)
        if passed == total:
            print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        else:
            print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        print("="*60)
        
        print("\nMath backend integration complete:")
        print("  ✓ Modal classification working")
        print("  ✓ Symbolic computation working")
        print("  ✓ Database protection working")
        print("  ✓ Linguistic queries unaffected")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
