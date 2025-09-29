import pytest

from dashboard import to_bool


@pytest.mark.parametrize(
    "value",
    [True, "true", "True", " true ", "1", "Yes", "y", "on", "enabled", "active", "hit", "triggered", "reached"],
)
def test_to_bool_truthy(value):
    assert to_bool(value) is True


@pytest.mark.parametrize(
    "value",
    [False, "false", "0", "no", "", None, 0],
)
def test_to_bool_falsy(value):
    assert to_bool(value) is False
