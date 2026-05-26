from stereomapper.domain.relationship_terms import get_relationship_term, get_relationship_term_id


def test_get_relationship_term_known_label():
    term = get_relationship_term("Enantiomers")
    assert term is not None
    assert term.term_id == "SMRO:0005"


def test_get_relationship_term_case_insensitive():
    assert get_relationship_term_id("stereo resolution pairs") == "SMRO:0007"


def test_get_relationship_term_unknown():
    assert get_relationship_term("Not a class") is None
    assert get_relationship_term_id("") is None
