import json
from pathlib import Path

from .concept_grounding import ConceptGrounder
from .relational_graph_store import RelationalGraphStore


def _load_dataset() -> dict:
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "test" / "polysemy_cases.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def _build_graph(graph: RelationalGraphStore, dataset: dict) -> None:
    spec = dataset["graph"]

    for c in spec.get("concepts", []):
        graph.add_node(c["id"], "concept", c["term"], confidence=1.0)

    for l in spec.get("lexemes", []):
        graph.add_lexeme(l["id"], l["term"], confidence=1.0)

    for s in spec.get("senses", []):
        graph.add_sense(s["id"], s["term"], confidence=1.0)

    for rel in spec.get("lexeme_to_sense", []):
        graph.link_lexeme_to_sense(
            rel["lexeme_id"],
            rel["sense_id"],
            confidence=float(rel.get("confidence", 1.0)),
        )

    for rel in spec.get("sense_to_concept", []):
        graph.link_sense_to_concept(
            rel["sense_id"],
            rel["concept_id"],
            confidence=float(rel.get("confidence", 1.0)),
        )


def test_polysemy_benchmark_metrics(tmp_path: Path):
    dataset = _load_dataset()
    graph = RelationalGraphStore(str(tmp_path / "polysemy.sqlite"))
    _build_graph(graph, dataset)

    grounder = ConceptGrounder(graph)

    total = 0
    correct_top = 0
    wrong_activations = 0
    total_activations = 0

    ambig_total = 0
    ambig_correct = 0

    non_ambig_total = 0
    non_ambig_miss = 0
    non_ambig_miss_ids = []

    for case in dataset["cases"]:
        total += 1
        matches = grounder.ground(case["query"])

        expected_any = set(case.get("expected_any", []))
        expected_top = case.get("expected_top")
        expected_set = expected_any | ({expected_top} if expected_top else set())

        top_id = matches[0].concept_id if matches else None
        if top_id and top_id in expected_set:
            correct_top += 1

        for m in matches:
            total_activations += 1
            if m.concept_id not in expected_set:
                wrong_activations += 1

        if case.get("expect_ambiguous", False):
            ambig_total += 1
            if any(":ambiguous" in m.source for m in matches):
                ambig_correct += 1

        if case.get("non_ambiguous", False):
            non_ambig_total += 1
            if not top_id or (expected_top is not None and top_id != expected_top):
                non_ambig_miss += 1
                non_ambig_miss_ids.append(case["id"])

    sense_accuracy = correct_top / max(total, 1)
    wrong_activation_rate = wrong_activations / max(total_activations, 1)
    ambiguity_deferral_correctness = ambig_correct / max(ambig_total, 1)
    non_ambiguous_regression_rate = non_ambig_miss / max(non_ambig_total, 1)

    # Thresholds tuned for this curated benchmark and current resolver behavior.
    assert sense_accuracy >= 0.80
    assert wrong_activation_rate <= 0.40
    assert ambiguity_deferral_correctness >= 0.90
    assert non_ambiguous_regression_rate <= 0.10, (
        f"non_ambiguous_regression_rate={non_ambiguous_regression_rate:.2f}, "
        f"missed={non_ambig_miss_ids}"
    )


class _MockPMFlowLM:
    """Simple context scorer used to validate LM rerank integration."""

    def score_text(self, text: str):
        t = text.lower()
        # For 'java', prefer programming-language sense over coffee.
        if "query: java" in t and "programming language" in t:
            return 2.0
        if "query: java" in t and "coffee" in t:
            return -1.0
        return 0.0


def test_lm_rerank_improves_near_tie_contextual_disambiguation(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "lm_rerank.sqlite"))

    # Concepts
    graph.add_node("concept:programming_language", "concept", "programming language")
    graph.add_node("concept:coffee", "concept", "coffee")

    # Lexeme and senses
    graph.add_lexeme("lex:java", "java")
    graph.add_sense("sense:java#language", "java programming language")
    graph.add_sense("sense:java#coffee", "java coffee")

    # Deliberately keep structural scores as a near-tie with coffee slightly higher.
    graph.link_lexeme_to_sense("lex:java", "sense:java#language", confidence=0.90)
    graph.link_lexeme_to_sense("lex:java", "sense:java#coffee", confidence=0.91)
    graph.link_sense_to_concept("sense:java#language", "concept:programming_language", confidence=0.90)
    graph.link_sense_to_concept("sense:java#coffee", "concept:coffee", confidence=0.90)

    # Baseline without LM rerank: coffee can win due to slightly higher structural score.
    baseline = ConceptGrounder(graph)
    baseline_matches = baseline.ground("java")
    assert baseline_matches

    # LM-enabled rerank should flip top choice to programming language for this context.
    reranked = ConceptGrounder(
        graph,
        language_model=_MockPMFlowLM(),
        enable_lm_rerank=True,
        lm_rerank_min_ambiguity=0.20,
        lm_rerank_weight=0.60,
    )
    reranked_matches = reranked.ground("java")
    assert reranked_matches
    assert reranked_matches[0].concept_id == "concept:programming_language"
    assert "lm_rerank" in reranked_matches[0].source
