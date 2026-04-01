from pathlib import Path

from .concept_grounding import ConceptGrounder
from .relational_graph_store import RelationalGraphStore


ARTIFACT_TYPES = {
    "lexical_token",
    "utterance_parsed",
    "symbolic_frame",
    "pos_pattern",
    "syntax_pattern",
    "fragment",
    "pattern",
}


def test_relational_graph_store_get_terms_by_types_filters(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))
    graph.add_node("c_python", "concept", "python")
    graph.add_node("tok_python_nn", "lexical_token", "python")
    graph.add_node("pat_python", "syntax_pattern", "python is a language")

    terms = graph.get_terms_by_types(
        allowed_types=["concept", "entity"],
        excluded_types=["lexical_token", "syntax_pattern"],
    )

    ids = {node_id for node_id, _ in terms}
    assert "c_python" in ids
    assert "tok_python_nn" not in ids
    assert "pat_python" not in ids


def test_concept_grounder_excludes_lexical_and_pattern_nodes(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    # Same surface term appears in both lexical and concept spaces.
    graph.add_node("c_bank_finance", "concept", "bank")
    graph.add_node("tok_bank_nn", "lexical_token", "bank")
    graph.add_node("pat_bank", "syntax_pattern", "bank account")

    grounder = ConceptGrounder(graph)
    matches = grounder.ground("bank")

    assert matches, "Expected at least one grounding match for 'bank'"

    matched_ids = {m.concept_id for m in matches}
    assert "c_bank_finance" in matched_ids
    assert "tok_bank_nn" not in matched_ids
    assert "pat_bank" not in matched_ids


def test_grounding_improvement_prevents_full_phrase_pattern_hijack(tmp_path: Path):
    """Regression test: full-phrase matches must not early-return syntax artifacts.

    Prior behavior could return a syntax-pattern node on full-phrase match and stop,
    preventing concept grounding. With type-filtered term cache, only semantic
    node types participate, so the same query should resolve to concept nodes.
    """
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    # Semantic target nodes.
    graph.add_node("c_python", "concept", "python")
    graph.add_node("c_language", "concept", "language")

    # Non-semantic node that used to hijack full-phrase grounding.
    graph.add_node("pat_def", "syntax_pattern", "python is a language")

    grounder = ConceptGrounder(graph)
    matches = grounder.ground("python is a language")

    assert matches, "Expected semantic matches for phrase grounding"
    ids = {m.concept_id for m in matches}

    # Improvement assertions:
    # 1) syntax pattern does not hijack grounding
    # 2) semantic concept(s) are returned instead
    assert "pat_def" not in ids
    assert "c_python" in ids or "c_language" in ids


def test_grounding_quality_benchmark_artifact_hit_rate_zero(tmp_path: Path):
    """Benchmark-style regression over multiple phrase/ambiguity cases.

    We treat these as quality metrics:
    - artifact_hit_rate should be 0.0
    - semantic_hit_rate should stay high for this curated set
    """
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    # Semantic nodes
    graph.add_node("c_python", "concept", "python")
    graph.add_node("c_language", "concept", "language")
    graph.add_node("c_bank_finance", "concept", "bank")
    graph.add_node("c_river_bank", "entity", "river bank")
    graph.add_node("c_java", "concept", "java")
    graph.add_node("c_programming", "concept", "programming")

    # Artifact nodes that should never be returned by grounding
    graph.add_node("pat_python", "syntax_pattern", "python is a language")
    graph.add_node("tok_bank_nn", "lexical_token", "bank")
    graph.add_node("pat_java", "pattern", "java programming")
    graph.add_node("utt_1", "utterance_parsed", "bank near river")

    cases = [
        "python is a language",
        "bank",
        "java programming",
        "bank near river",
    ]

    grounder = ConceptGrounder(graph)

    artifact_hits = 0
    semantic_hits = 0
    total_matches = 0

    for text in cases:
        matches = grounder.ground(text)
        if not matches:
            continue
        for m in matches:
            node = graph.get_node(m.concept_id)
            if not node:
                continue
            total_matches += 1
            node_type = node.get("type")
            if node_type in ARTIFACT_TYPES:
                artifact_hits += 1
            else:
                semantic_hits += 1

    assert total_matches > 0, "Benchmark produced no matches"

    artifact_hit_rate = artifact_hits / total_matches
    semantic_hit_rate = semantic_hits / total_matches

    assert artifact_hit_rate == 0.0
    assert semantic_hit_rate >= 0.75


def test_grounder_resolves_lexeme_to_concepts_via_senses(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    graph.add_lexeme("lex_bank", "bank")
    graph.add_sense("sense_bank_finance", "bank (financial institution)")
    graph.add_sense("sense_bank_river", "bank (river edge)")
    graph.add_node("concept_financial_institution", "concept", "financial institution")
    graph.add_node("concept_river_edge", "concept", "river edge")

    graph.link_lexeme_to_sense("lex_bank", "sense_bank_finance", confidence=0.9)
    graph.link_lexeme_to_sense("lex_bank", "sense_bank_river", confidence=0.85)
    graph.link_sense_to_concept("sense_bank_finance", "concept_financial_institution", confidence=0.95)
    graph.link_sense_to_concept("sense_bank_river", "concept_river_edge", confidence=0.92)

    grounder = ConceptGrounder(graph)
    matches = grounder.ground("bank")
    ids = {m.concept_id for m in matches}

    assert "concept_financial_institution" in ids
    assert "concept_river_edge" in ids
    assert all(m.sense_id is not None for m in matches)


def test_grounder_resolves_direct_sense_node_to_concept(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    graph.add_sense("sense_java_lang", "java")
    graph.add_node("concept_java_language", "concept", "java language")
    graph.link_sense_to_concept("sense_java_lang", "concept_java_language", confidence=0.9)

    grounder = ConceptGrounder(graph)
    matches = grounder.ground("java")

    assert matches
    top = matches[0]
    assert top.concept_id == "concept_java_language"
    assert top.sense_id == "sense_java_lang"
    assert "sense_to_concept" in top.source or "lexeme_to_sense_to_concept" in top.source


def test_grounding_trace_includes_score_breakdown_fields(tmp_path: Path):
    class MockLM:
        def score_text(self, text: str):
            t = text.lower()
            if "programming language" in t:
                return 1.0
            return -0.5

    graph = RelationalGraphStore(str(tmp_path / "trace.sqlite"))
    graph.add_node("concept:programming_language", "concept", "programming language")
    graph.add_node("concept:coffee", "concept", "coffee")
    graph.add_lexeme("lex:java", "java")
    graph.add_sense("sense:java#language", "java programming language")
    graph.add_sense("sense:java#coffee", "java coffee")
    graph.link_lexeme_to_sense("lex:java", "sense:java#language", confidence=0.90)
    graph.link_lexeme_to_sense("lex:java", "sense:java#coffee", confidence=0.91)
    graph.link_sense_to_concept("sense:java#language", "concept:programming_language", confidence=0.90)
    graph.link_sense_to_concept("sense:java#coffee", "concept:coffee", confidence=0.90)

    grounder = ConceptGrounder(
        graph,
        language_model=MockLM(),
        enable_lm_rerank=True,
        lm_rerank_min_ambiguity=0.20,
        lm_rerank_weight=0.60,
    )

    _ = grounder.ground("java")
    trace = grounder.get_last_trace()

    assert trace.get("query") == "java"
    assert isinstance(trace.get("resolved_candidates"), list)
    assert trace["resolved_candidates"], "Expected resolved candidates in trace"

    first = trace["resolved_candidates"][0]
    assert "structural_score" in first
    assert "lm_score" in first
    assert "fused_score" in first
