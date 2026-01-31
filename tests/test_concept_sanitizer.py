from lilith.concept_sanitizer import sanitize_properties


def test_sanitize_properties_filters_long_dupe_and_term_miss():
    term = "bird"
    props = [
        "birds have feathers",
        "birds have feathers",  # duplicate
        "this is unrelated text",
        "a" * 300,  # too long
    ]
    cleaned, stats = sanitize_properties(term, props, max_len=50, require_term=True, dedupe=True)

    assert "birds have feathers" in cleaned
    assert len(cleaned) == 1  # duplicates and unrelated removed
    assert stats["removed_dupe"] == 1
    assert stats["removed_long"] == 1
    assert stats["removed_term"] == 1
