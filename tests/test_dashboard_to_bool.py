import pytest

from dashboard import to_bool


@pytest.mark.parametrize(
    "value",
    [
        True,
        "true",
        "True",
        " true ",
        "1",
        "Yes",
        "y",
        "on",
        "enabled",
        "active",
        "hit",
        " Hit ",
        "triggered",
        " triggered ",
        "reached",
        "target hit",
        "TARGET REACHED",
        "tp1_hit",
        "EXECUTED",
        "done",
    ],
)
def test_to_bool_truthy(value):
    assert to_bool(value) is True


@pytest.mark.parametrize(
    "value",
    [False, "false", "0", "no", "", None, 0, "pending", " waiting "],
)
def test_to_bool_falsy(value):
    assert to_bool(value) is False
