import sqlite3
from pathlib import Path

from lilith.relational_concept_store import RelationalConceptStore


def _setup_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "concepts.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE concepts (
            concept_id TEXT PRIMARY KEY,
            term TEXT NOT NULL,
            confidence REAL DEFAULT 0.85,
            source TEXT DEFAULT 'taught',
            usage_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_id TEXT NOT NULL,
            property_text TEXT NOT NULL
        );
        CREATE TABLE relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            target TEXT NOT NULL,
            confidence REAL DEFAULT 0.85
        );
        """
    )
    cur.execute("INSERT INTO concepts(concept_id, term) VALUES (?, ?)", ("concept_1", "bird"))
    cur.execute("INSERT INTO concepts(concept_id, term) VALUES (?, ?)", ("concept_2", "vertebrate"))
    cur.execute("INSERT INTO properties(concept_id, property_text) VALUES (?, ?)", ("concept_1", "birds have feathers"))
    cur.execute(
        "INSERT INTO relations(concept_id, relation_type, target, confidence) VALUES (?, ?, ?, ?)",
        ("concept_1", "is_type_of", "vertebrate", 0.9),
    )
    cur.execute(
        "INSERT INTO relations(concept_id, relation_type, target, confidence) VALUES (?, ?, ?, ?)",
        ("concept_2", "has_subtype", "bird", 0.9),
    )
    conn.commit()
    conn.close()
    return db_path


def test_relational_queries(tmp_path):
    db_path = _setup_db(tmp_path)
    store = RelationalConceptStore(str(db_path))

    concepts = store.query_concepts("bird", limit=5)
    assert concepts, "expected at least one concept match"
    assert concepts[0]["term"].lower().startswith("bird")

    props = store.query_properties("bird", limit=3)
    assert "birds have feathers" in props

    rels = store.query_relations("bird", relation_type="is_type_of", limit=3)
    assert any(r["target"] == "vertebrate" for r in rels)

    chains = store.two_hop_chains("bird", limit=5)
    assert any(ch["final_target"] == "bird" for ch in chains)

    store.close()
