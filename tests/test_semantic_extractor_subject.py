from lilith.semantic_extractor import SemanticExtractor


def test_extract_subject_strips_pronouns_and_punctuation():
    ex = SemanticExtractor()

    assert ex._extract_subject_from_query("you fog?") == "Fog"
    assert ex._extract_subject_from_query("can you explain fog?") == "Fog"
    assert ex._extract_subject_from_query("what do you know about fog?") == "Fog"


def test_extract_subject_preserves_acronyms_like_it():
    ex = SemanticExtractor()

    assert ex._extract_subject_from_query("what is IT security?") == "IT security"
