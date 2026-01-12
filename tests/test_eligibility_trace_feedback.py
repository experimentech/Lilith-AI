from __future__ import annotations


def test_weighted_feedback_applies_to_all_fragments(tmp_path):
    from lilith.session import LilithSession, SessionConfig

    cfg = SessionConfig(
        data_path=str(tmp_path),
        learning_enabled=True,
        enable_feedback_detection=False,
        enable_auto_learning=False,
        enable_declarative_learning=False,
        enable_mood=False,
        enable_personality=False,
        enable_preferences=False,
        enable_world_model=False,
        enable_reasoning=False,
        enable_knowledge_augmentation=False,
        enable_modal_routing=False,
        use_grammar=False,
        enable_compositional=False,
        enable_pragmatic_templates=False,
        composition_mode="best_match",
    )

    session = LilithSession(user_id="elig-test", context_id="test", config=cfg)

    # Create two user patterns that will be eligible for reinforcement.
    pid_a = session.store.add_pattern("alpha trigger", "alpha response", success_score=0.5, intent="test")
    pid_b = session.store.add_pattern("beta trigger", "beta response", success_score=0.5, intent="test")

    # Seed an eligibility record indicating both fragments contributed.
    session._eligibility_buffer.append(
        {
            "user_input": "some user prompt",
            "response_text": "composed response",
            "fragment_ids": [pid_a, pid_b],
            "weights": [0.8, 0.2],
        }
    )

    def _score(fragment_id: str) -> float:
        user_store = session.store.user_store
        assert user_store is not None
        conn = user_store._get_connection()
        row = conn.execute(
            "SELECT success_score FROM response_patterns WHERE fragment_id = ?",
            (fragment_id,),
        ).fetchone()
        conn.close()
        assert row is not None
        return float(row[0])

    a_before = _score(pid_a)
    b_before = _score(pid_b)

    session.upvote(pid_a, strength=0.4)

    a_after = _score(pid_a)
    b_after = _score(pid_b)

    assert a_after > a_before
    assert b_after > b_before
    # Heavier weight should receive larger increase.
    assert (a_after - a_before) > (b_after - b_before)
