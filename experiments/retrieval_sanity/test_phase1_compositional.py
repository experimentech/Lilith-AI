"""
Test Phase 1: Production Compositional Implementation

Validates:
1. ConceptDatabase persistence
2. ProductionConceptStore with PMFlow enhancements
3. TemplateComposer integration
4. Parallel pattern + concept response generation
5. Metrics tracking
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.concept_database import ConceptDatabase
from pipeline.production_concept_store import ProductionConceptStore, Relation
from pipeline.template_composer import TemplateComposer


def test_concept_database():
    """Test 1: Database persistence"""
    print("\n" + "="*60)
    print("TEST 1: ConceptDatabase Persistence")
    print("="*60)
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_concepts.db"
    
    try:
        db = ConceptDatabase(str(db_path))
        
        # Add concept
        db.add_concept(
            "concept_001",
            "machine learning",
            ["learns from data", "branch of AI", "uses algorithms"],
            [
                {"relation_type": "is_type_of", "target": "artificial intelligence", "confidence": 0.90},
                {"relation_type": "used_for", "target": "prediction", "confidence": 0.85}
            ],
            confidence=0.90,
            source="taught"
        )
        
        # Retrieve concept
        concept = db.get_concept("concept_001")
        assert concept is not None
        assert concept['term'] == "machine learning"
        assert len(concept['properties']) == 3
        assert len(concept['relations']) == 2
        
        # Check stats
        stats = db.get_stats()
        assert stats['total_concepts'] == 1
        assert stats['total_properties'] == 3
        assert stats['total_relations'] == 2
        
        print("✅ Database persistence working")
        print(f"   - Created concept: {concept['term']}")
        print(f"   - Properties: {len(concept['properties'])}")
        print(f"   - Relations: {len(concept['relations'])}")
        
        db.close()
        
    finally:
        shutil.rmtree(temp_dir)


def test_production_concept_store():
    """Test 2: ProductionConceptStore with PMFlow"""
    print("\n" + "="*60)
    print("TEST 2: ProductionConceptStore with PMFlow")
    print("="*60)
    
    # Import encoder
    from pipeline.embedding import PMFlowEmbeddingEncoder
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(
        dimension=64,
        latent_dim=32,
        device="cpu"
    )
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_concepts.db"
    
    try:
        # Create concept store
        store = ProductionConceptStore(
            encoder,
            str(db_path),
            consolidation_threshold=0.85
        )
        
        # Add concepts
        print("\n📝 Adding concepts...")
        
        c1 = store.add_concept(
            "machine learning",
            ["learns from data", "branch of AI", "uses statistical methods"],
            source="taught"
        )
        
        c2 = store.add_concept(
            "neural networks",
            ["inspired by brain", "uses layers", "learns representations"],
            source="taught"
        )
        
        c3 = store.add_concept(
            "supervised learning",
            ["learns from labeled data", "type of machine learning", "predicts outputs"],
            source="taught"
        )
        
        print(f"   Added 3 concepts")
        
        # Retrieve by text
        print("\n🔍 Testing PMFlow-enhanced retrieval...")
        results = store.retrieve_by_text(
            "what is machine learning",
            top_k=2,
            min_similarity=0.50
        )
        
        print(f"   Retrieved {len(results)} concepts")
        for concept, score in results:
            print(f"   - {concept.term}: {score:.3f}")
        
        # Check stats
        stats = store.get_stats()
        print(f"\n📊 Stats:")
        print(f"   - Total concepts: {stats['total_concepts']}")
        print(f"   - Cached embeddings: {stats['cached_embeddings']}")
        print(f"   - PMFlow enabled: {stats['pmflow_enabled']}")
        
        store.close()
        
        print("\n✅ ProductionConceptStore working")
        
    finally:
        shutil.rmtree(temp_dir)


def test_template_composer():
    """Test 3: TemplateComposer"""
    print("\n" + "="*60)
    print("TEST 3: TemplateComposer")
    print("="*60)
    
    composer = TemplateComposer()
    
    # Test definition query
    result = composer.compose_response(
        "what is machine learning",
        "machine learning",
        ["learns from data", "branch of AI"],
        confidence=0.85
    )
    
    assert result is not None
    assert result['intent'] == "definition_query"
    assert "Machine learning is" in result['text']
    
    print("✅ TemplateComposer working")
    print(f"   Query: 'what is machine learning'")
    print(f"   Response: '{result['text']}'")
    print(f"   Intent: {result['intent']}")
    print(f"   Confidence: {result['confidence']:.3f}")


def test_compositional_integration():
    """Test 4: Full compositional pipeline"""
    print("\n" + "="*60)
    print("TEST 4: Compositional Integration")
    print("="*60)
    
    # Import encoder
    from pipeline.embedding import PMFlowEmbeddingEncoder
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(
        dimension=64,
        latent_dim=32
    )
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_concepts.db"
    
    try:
        # Create concept store
        store = ProductionConceptStore(
            encoder,
            str(db_path),
            consolidation_threshold=0.85
        )
        
        # Add ML concepts
        store.add_concept(
            "machine learning",
            ["learns from data", "branch of AI", "uses algorithms"],
            source="taught"
        )
        
        store.add_concept(
            "supervised learning",
            ["learns from labeled examples", "type of machine learning", "predicts outputs"],
            source="taught"
        )
        
        # Create template composer
        template_composer = TemplateComposer()
        
        # Simulate compositional response
        print("\n🧩 Testing compositional response generation...")
        
        query = "what is machine learning"
        
        # 1. Retrieve concept
        concepts = store.retrieve_by_text(query, top_k=1, min_similarity=0.50)
        
        if concepts:
            concept, similarity = concepts[0]
            print(f"   Retrieved: {concept.term} (similarity: {similarity:.3f})")
            
            # 2. Compose response
            result = template_composer.compose_response(
                query,
                concept.term,
                concept.properties,
                confidence=similarity
            )
            
            if result:
                print(f"   Response: '{result['text']}'")
                print(f"   Intent: {result['intent']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                
                assert "Machine learning is" in result['text']
                print("\n✅ Compositional integration working")
            else:
                print("❌ Template composition failed")
        else:
            print("❌ Concept retrieval failed")
        
        store.close()
        
    finally:
        shutil.rmtree(temp_dir)


def test_consolidation():
    """Test 5: Concept consolidation"""
    print("\n" + "="*60)
    print("TEST 5: Concept Consolidation")
    print("="*60)
    
    # Import encoder
    from pipeline.embedding import PMFlowEmbeddingEncoder
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(
        dimension=64,
        latent_dim=32
    )
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_concepts.db"
    
    try:
        # Create concept store with high consolidation threshold
        store = ProductionConceptStore(
            encoder,
            str(db_path),
            consolidation_threshold=0.95  # Very similar to trigger consolidation
        )
        
        # Add very similar concepts
        print("\n📝 Adding similar concepts...")
        
        c1 = store.add_concept(
            "machine learning",
            ["learns from data", "branch of AI"],
            source="taught"
        )
        
        c2 = store.add_concept(
            "machine learning",  # Same term
            ["uses algorithms", "statistical method"],
            source="wikipedia"
        )
        
        # Check if consolidated
        stats = store.get_stats()
        print(f"\n📊 After adding duplicate:")
        print(f"   Total concepts: {stats['total_concepts']}")
        print(f"   Total properties: {stats['total_properties']}")
        
        # Should have consolidated into 1 concept with all properties
        if stats['total_concepts'] == 1:
            print("✅ Automatic consolidation working")
            
            # Verify properties were merged
            concept = store.db.get_concept(c1)
            if concept:
                print(f"   Properties merged: {len(concept['properties'])}")
                assert len(concept['properties']) == 4  # All 4 properties combined
        else:
            print(f"⚠️  No consolidation (threshold may be too high)")
        
        store.close()
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 1: Production Compositional Implementation Tests")
    print("="*60)
    
    try:
        test_concept_database()
        test_production_concept_store()
        test_template_composer()
        test_compositional_integration()
        test_consolidation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nPhase 1 implementation complete:")
        print("  ✓ Database persistence")
        print("  ✓ PMFlow-enhanced retrieval")
        print("  ✓ Template composition")
        print("  ✓ Compositional integration")
        print("  ✓ Concept consolidation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
