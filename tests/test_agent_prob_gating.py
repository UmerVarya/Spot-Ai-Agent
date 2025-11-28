from config import DEFAULT_MIN_PROB_FOR_TRADE
from probability_gating import should_veto_on_probability


def test_probability_veto_matches_shared_threshold():
    below_threshold = DEFAULT_MIN_PROB_FOR_TRADE - 0.01
    above_threshold = DEFAULT_MIN_PROB_FOR_TRADE + 0.01

    assert should_veto_on_probability(below_threshold) is True
    assert should_veto_on_probability(above_threshold) is False

