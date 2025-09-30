from dashboard import to_bool


def test_to_bool_detects_negated_words():
    assert to_bool("Not triggered") is False
    assert to_bool("tp1_not_hit") is False
    assert to_bool("SL not reached") is False
    assert to_bool("pending hit") is False


def test_to_bool_positive_keywords_without_negation():
    assert to_bool("tp hit") is True
    assert to_bool("target reached") is True
