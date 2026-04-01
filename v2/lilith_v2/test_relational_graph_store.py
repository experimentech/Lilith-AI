from pathlib import Path

from .relational_graph_store import RelationalGraphStore


def test_sense_layer_links_lexeme_to_sense_to_concept(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    graph.add_lexeme("lex_bank", "bank")
    graph.add_sense("sense_bank_finance", "bank (financial institution)")
    graph.add_sense("sense_bank_river", "bank (river edge)")
    graph.add_node("concept_financial_institution", "concept", "financial institution")
    graph.add_node("concept_river_edge", "concept", "river edge")

    graph.link_lexeme_to_sense("lex_bank", "sense_bank_finance", relation="possible_meaning", confidence=0.8)
    graph.link_lexeme_to_sense("lex_bank", "sense_bank_river", relation="possible_meaning", confidence=0.7)

    graph.link_sense_to_concept("sense_bank_finance", "concept_financial_institution", confidence=0.9)
    graph.link_sense_to_concept("sense_bank_river", "concept_river_edge", confidence=0.9)

    senses = graph.get_senses_for_lexeme("lex_bank")
    assert len(senses) == 2
    assert {s["id"] for s in senses} == {"sense_bank_finance", "sense_bank_river"}

    mapped = graph.get_concepts_for_sense("sense_bank_finance")
    assert len(mapped) == 1
    assert mapped[0]["id"] == "concept_financial_institution"
    assert mapped[0]["relation"] == "maps_to"


def test_sense_nodes_are_available_to_type_filtered_term_queries(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    graph.add_sense("sense_run_motion", "run (move quickly)")
    graph.add_node("tok_run_vb", "lexical_token", "run")

    terms = graph.get_terms_by_types(allowed_types=["sense", "concept"])
    ids = {node_id for node_id, _ in terms}

    assert "sense_run_motion" in ids
    assert "tok_run_vb" not in ids


def test_get_or_create_concept_reuses_stable_id_for_same_term(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    c1 = graph.get_or_create_concept("Python", confidence=0.8)
    c2 = graph.get_or_create_concept("python", confidence=0.9)

    assert c1 == c2
    node = graph.get_node(c1)
    assert node is not None
    assert node["type"] == "concept"
    assert node["term"].lower() == "python"


def test_get_or_create_concept_tracks_aliases(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    cid = graph.get_or_create_concept("financial institution", alias="bank")
    cid_again = graph.get_or_create_concept("Financial Institution", alias="depository bank")

    assert cid == cid_again
    node = graph.get_node(cid)
    aliases = set((node or {}).get("data", {}).get("aliases", []))
    assert "bank" in aliases
    assert "depository bank" in aliases


def test_layered_feedback_updates_lexeme_faster_than_concept(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    graph.add_lexeme("lex_run", "run", confidence=0.5)
    graph.add_node("concept:run_motion", "concept", "run motion", confidence=0.5)

    lex_after = graph.apply_feedback_to_node("lex_run", feedback_score=1.0)
    concept_after = graph.apply_feedback_to_node("concept:run_motion", feedback_score=1.0)

    assert lex_after is not None and concept_after is not None
    assert lex_after > concept_after
    assert concept_after <= 0.56  # Conservative concept reinforcement


def test_verified_sense_protection_and_pruning_guardrail(tmp_path: Path):
    graph = RelationalGraphStore(str(tmp_path / "graph.sqlite"))

    graph.add_sense("sense:bank_finance", "bank (financial)", confidence=0.30)
    graph.add_node("concept:financial_institution", "concept", "financial institution", confidence=0.8)
    graph.link_sense_to_concept("sense:bank_finance", "concept:financial_institution", confidence=0.9)

    # Apply strong negative feedback; verified sense should not collapse below protection floor.
    for _ in range(5):
        graph.apply_feedback_to_node("sense:bank_finance", feedback_score=-1.0)

    sense = graph.get_node("sense:bank_finance")
    assert sense is not None
    assert float(sense["confidence"]) >= 0.35

    # Even if below threshold, pruning should preserve verified senses.
    demoted = graph.prune_low_confidence_senses(threshold=0.4, preserve_verified=True)
    assert demoted == 0
    sense2 = graph.get_node("sense:bank_finance")
    assert sense2 is not None
    assert sense2.get("data", {}).get("status") != "dormant"
