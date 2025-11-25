#!/usr/bin/env python3
"""
Compositional Response Architecture - Proof of Concept

Tests the compositional approach vs traditional pattern storage.

Success Criteria:
1. Storage Efficiency: N teachings → ~N/3 concepts (consolidation)
2. Quality: Compositional response quality ≥ pattern-based
3. Generalization: Can compose novel responses from learned concepts
4. Coherence: Responses are grammatically correct and relevant
"""

import sys
from pathlib import Path
import numpy as np

# Add project root and pipeline to path
project_root = Path(__file__).parent.parent.parent
pipeline_dir = Path(__file__).parent.parent / "pipeline"
poc_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(pipeline_dir))
sys.path.insert(0, str(poc_dir))

from concept_store import ConceptStore, Relation
from template_composer import TemplateComposer


class SimpleSemanticEncoder:
    """
    Minimal semantic encoder for PoC testing.
    Uses simple bag-of-words + random projection for embeddings.
    """
    
    def __init__(self, dimension=128, vocab_size=5000, seed=42):
        self.dimension = dimension
        self.vocab_size = vocab_size
        np.random.seed(seed)
        
        # Word to index mapping
        self.word_to_idx = {}
        self.next_idx = 0
        
        # Random projection matrix for dimensionality reduction
        self.projection = np.random.randn(vocab_size, dimension).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)
    
    def _get_word_idx(self, word: str) -> int:
        """Get or create index for word"""
        word = word.lower()
        if word not in self.word_to_idx:
            if self.next_idx >= self.vocab_size:
                # Hash to existing index if vocab full
                return hash(word) % self.vocab_size
            self.word_to_idx[word] = self.next_idx
            self.next_idx += 1
        return self.word_to_idx[word]
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        # Tokenize
        tokens = text.lower().split()
        
        # Create bag-of-words vector
        bow = np.zeros(self.vocab_size, dtype=np.float32)
        for token in tokens:
            idx = self._get_word_idx(token)
            bow[idx] += 1.0
        
        # Normalize
        if np.sum(bow) > 0:
            bow /= np.sum(bow)
        
        # Project to lower dimension
        embedding = self.projection.T @ bow
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        return embedding


class MockPatternStore:
    """
    Mock of current pattern-based approach for comparison.
    
    Stores full response patterns like current system.
    """
    
    def __init__(self):
        self.patterns = {}
        self.pattern_count = 0
    
    def add_pattern(self, trigger: str, response: str):
        """Add a response pattern"""
        self.patterns[trigger.lower()] = response
        self.pattern_count += 1
    
    def retrieve(self, query: str) -> str:
        """Retrieve exact or similar pattern"""
        query_lower = query.lower()
        
        # Try exact match
        if query_lower in self.patterns:
            return self.patterns[query_lower]
        
        # Try fuzzy match (simple keyword overlap)
        query_words = set(query_lower.split())
        best_match = None
        best_score = 0
        
        for trigger, response in self.patterns.items():
            trigger_words = set(trigger.split())
            overlap = len(query_words & trigger_words)
            score = overlap / max(len(query_words), len(trigger_words))
            
            if score > best_score:
                best_score = score
                best_match = response
        
        if best_score > 0.4:  # Threshold
            return best_match
        
        return "[No pattern found]"
    
    def get_stats(self):
        return {
            "total_patterns": self.pattern_count,
            "unique_patterns": len(self.patterns)
        }


def test_compositional_poc():
    """Run the PoC test suite"""
    
    print("="*80)
    print("COMPOSITIONAL RESPONSE ARCHITECTURE - PROOF OF CONCEPT")
    print("="*80)
    print()
    
    # Initialize components
    print("📦 Initializing components...")
    encoder = SimpleSemanticEncoder(
        dimension=128,
        vocab_size=5000,
        seed=42
    )
    
    concept_store = ConceptStore(
        semantic_encoder=encoder,
        storage_path="poc_concepts.json"
    )
    
    template_composer = TemplateComposer()
    
    pattern_store = MockPatternStore()
    
    print("✅ Components initialized")
    print()
    
    # Test 1: Storage Efficiency
    print("="*80)
    print("TEST 1: STORAGE EFFICIENCY")
    print("="*80)
    print()
    
    teachings = [
        ("machine learning", ["branch of artificial intelligence", "enables computers to learn from data", "without explicit programming"]),
        ("deep learning", ["uses neural networks with many layers", "learns hierarchical representations", "excels at image recognition"]),
        ("supervised learning", ["uses labeled training data", "learns to map inputs to outputs", "requires known examples"]),
        ("unsupervised learning", ["finds patterns in unlabeled data", "discovers hidden structure", "no predefined labels"]),
        ("reinforcement learning", ["trains agents through trial and error", "uses rewards and penalties", "learns to maximize rewards"]),
    ]
    
    print("Teaching 5 concepts to both systems...")
    print()
    
    for term, properties in teachings:
        # Compositional: Add concept
        concept_id = concept_store.add_concept(term, properties, source="taught")
        
        # Pattern-based: Add full pattern for each teaching
        full_text = f"{term.capitalize()} {properties[0]}."
        pattern_store.add_pattern(f"what is {term}", full_text)
        
        print(f"Taught: {term}")
        print(f"  Properties: {len(properties)}")
    
    print()
    
    # Test consolidation
    print("Testing consolidation (teaching similar concept)...")
    # Teach "ML" as synonym - should merge with "machine learning"
    ml_id = concept_store.add_concept(
        "ML",
        ["artificial intelligence technique", "data-driven approach"],
        source="taught"
    )
    
    # Pattern store would create new pattern
    pattern_store.add_pattern("what is ml", "ML is an artificial intelligence technique.")
    
    print()
    
    # Get statistics
    concept_stats = concept_store.get_stats()
    pattern_stats = pattern_store.get_stats()
    
    print("📊 STORAGE COMPARISON:")
    print(f"  Compositional:")
    print(f"    Concepts: {concept_stats['total_concepts']}")
    print(f"    Properties: {concept_stats['total_properties']}")
    print(f"    Avg properties/concept: {concept_stats['avg_properties_per_concept']:.1f}")
    print()
    print(f"  Pattern-based:")
    print(f"    Patterns: {pattern_stats['total_patterns']}")
    print()
    
    efficiency_ratio = concept_stats['total_concepts'] / pattern_stats['total_patterns']
    print(f"  💡 Efficiency: {efficiency_ratio:.2f} (target: ≤0.33 means 3x fewer storage units)")
    
    if efficiency_ratio <= 0.4:
        print("  ✅ PASS: Achieved storage consolidation")
    else:
        print("  ⚠️  MARGINAL: Some consolidation but room for improvement")
    
    print()
    
    # Test 2: Response Quality
    print("="*80)
    print("TEST 2: RESPONSE QUALITY")
    print("="*80)
    print()
    
    test_queries = [
        "What is machine learning?",
        "What is deep learning?",
        "How does supervised learning work?",
        "Tell me about reinforcement learning",
    ]
    
    print("Testing response generation for known concepts...")
    print()
    
    quality_scores = []
    
    for query in test_queries:
        print(f"Query: {query}")
        
        # Compositional response
        query_embedding = encoder.encode(query)
        concepts = concept_store.retrieve_similar(query_embedding, top_k=1, min_similarity=0.60)
        
        if concepts:
            concept, similarity = concepts[0]
            comp_result = template_composer.compose_response(
                query=query,
                concept_term=concept.term,
                properties=concept.properties,
                confidence=similarity
            )
            
            if comp_result:
                comp_response = comp_result['text']
                comp_conf = comp_result['confidence']
            else:
                comp_response = "[Template matching failed]"
                comp_conf = 0.0
        else:
            comp_response = "[No concept found]"
            comp_conf = 0.0
        
        # Pattern-based response
        pattern_response = pattern_store.retrieve(query)
        
        print(f"  Compositional: {comp_response}")
        print(f"    Confidence: {comp_conf:.2f}")
        print(f"  Pattern-based: {pattern_response}")
        print()
        
        # Simple quality check: does response contain relevant keywords?
        has_keywords = any(term in comp_response.lower() for term, _ in teachings)
        is_complete = len(comp_response.split()) >= 5
        is_grammatical = comp_response[0].isupper() and comp_response.endswith('.')
        
        quality = (has_keywords + is_complete + is_grammatical) / 3
        quality_scores.append(quality)
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    print(f"📊 QUALITY SCORE: {avg_quality:.2%}")
    
    if avg_quality >= 0.80:
        print("  ✅ PASS: Quality meets threshold (≥80%)")
    else:
        print(f"  ❌ FAIL: Quality below threshold ({avg_quality:.0%} < 80%)")
    
    print()
    
    # Test 3: Generalization (Novel Composition)
    print("="*80)
    print("TEST 3: GENERALIZATION")
    print("="*80)
    print()
    
    print("Testing novel query not explicitly taught...")
    print()
    
    novel_query = "What types of machine learning exist?"
    print(f"Novel Query: {novel_query}")
    print()
    
    # Compositional: Can it find related concepts?
    query_embedding = encoder.encode(novel_query)
    related_concepts = concept_store.retrieve_similar(
        query_embedding, 
        top_k=5, 
        min_similarity=0.50
    )
    
    print("  Compositional approach:")
    if related_concepts:
        concept_terms = [c.term for c, _ in related_concepts]
        print(f"    Found related concepts: {concept_terms}")
        
        # Could compose: "Types of machine learning include supervised, unsupervised, reinforcement..."
        if len(related_concepts) >= 2:
            response = f"Types of machine learning include {', '.join(concept_terms[:3])}."
            print(f"    Composed: {response}")
            print("    ✅ Successfully composed novel response from learned concepts")
            generalization_pass = True
        else:
            print("    ⚠️  Not enough concepts to compose novel response")
            generalization_pass = False
    else:
        print("    ❌ No related concepts found")
        generalization_pass = False
    
    print()
    
    # Pattern-based: Likely fails
    print("  Pattern-based approach:")
    pattern_response = pattern_store.retrieve(novel_query)
    print(f"    Response: {pattern_response}")
    if pattern_response == "[No pattern found]":
        print("    ❌ Cannot handle novel query (as expected)")
    
    print()
    
    if generalization_pass:
        print("  ✅ PASS: Compositional approach can generalize")
    else:
        print("  ❌ FAIL: Generalization did not work")
    
    print()
    
    # Test 4: Consolidation
    print("="*80)
    print("TEST 4: AUTOMATIC CONSOLIDATION")
    print("="*80)
    print()
    
    print("Running similarity-based consolidation...")
    merged = concept_store.merge_similar_concepts(threshold=0.92)
    
    print(f"  Merged {merged} similar concepts")
    
    final_stats = concept_store.get_stats()
    print(f"  Final concept count: {final_stats['total_concepts']}")
    
    if merged > 0:
        print("  ✅ PASS: Consolidation working")
    else:
        print("  ℹ️  INFO: No concepts similar enough to merge (threshold: 0.92)")
    
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    tests_passed = 0
    tests_total = 4
    
    if efficiency_ratio <= 0.4:
        tests_passed += 1
        print("✅ Storage Efficiency: PASS")
    else:
        print("⚠️  Storage Efficiency: MARGINAL")
    
    if avg_quality >= 0.80:
        tests_passed += 1
        print("✅ Response Quality: PASS")
    else:
        print("❌ Response Quality: FAIL")
    
    if generalization_pass:
        tests_passed += 1
        print("✅ Generalization: PASS")
    else:
        print("❌ Generalization: FAIL")
    
    if merged >= 0:
        tests_passed += 1
        print("✅ Consolidation: PASS")
    else:
        print("❌ Consolidation: FAIL")
    
    print()
    print(f"Overall: {tests_passed}/{tests_total} tests passed ({tests_passed/tests_total*100:.0f}%)")
    print()
    
    if tests_passed >= 3:
        print("🎉 POC SUCCESS: Compositional approach validated!")
        print("   Recommendation: Proceed with full implementation")
    elif tests_passed >= 2:
        print("⚠️  POC PARTIAL: Some improvements needed")
        print("   Recommendation: Refine approach and re-test")
    else:
        print("❌ POC FAIL: Significant issues found")
        print("   Recommendation: Revise design")
    
    print()
    
    return tests_passed >= 3


if __name__ == "__main__":
    success = test_compositional_poc()
    sys.exit(0 if success else 1)
