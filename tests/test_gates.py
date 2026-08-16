import pytest

from hiringcue import gates


def numeric_gate(required=3, operator=">="):
    return {
        "gate_id": "HG1",
        "requirement": f"At least {required} years",
        "operator": operator,
        "required_value": required,
        "unit": "years",
        "minimum_margin_unit": 1,
    }


def categorical_gate(required="demonstrated"):
    return {
        "gate_id": "HG2",
        "requirement": "Demonstrated knowledge",
        "operator": "==",
        "required_value": required,
        "unit": None,
        "minimum_margin_unit": "one evidence state",
    }


@pytest.mark.parametrize(
    "candidate,expected,margin",
    [(2, False, -1.0), (3, True, 0.0), (5, True, 2.0)],
)
def test_numeric_gate_margin_is_signed_distance(candidate, expected, margin):
    result = gates.evaluate_gate(numeric_gate(), candidate)
    assert result.passed is expected
    assert result.numeric_margin == margin
    assert result.margin_kind == "numeric"


def test_categorical_gate_is_case_and_whitespace_insensitive():
    assert gates.evaluate_gate(categorical_gate(), "  Demonstrated ").passed


def test_categorical_gate_has_no_numeric_margin():
    result = gates.evaluate_gate(categorical_gate(), "demonstrated")
    assert result.numeric_margin is None


def test_unsupported_operator_stops_rather_than_guessing():
    gate = numeric_gate()
    gate["operator"] = "approximately"
    with pytest.raises(gates.GateError):
        gates.evaluate_gate(gate, 3)


def test_non_numeric_value_on_numeric_operator_raises():
    with pytest.raises(gates.GateError):
        gates.evaluate_gate(numeric_gate(), "three")


def test_gold_decision_is_the_conjunction():
    hard = [numeric_gate(), categorical_gate()]
    passing = gates.evaluate_all(
        hard,
        [
            {"gate_id": "HG1", "candidate_value": 4},
            {"gate_id": "HG2", "candidate_value": "demonstrated"},
        ],
    )
    assert gates.gold_decision(passing) == gates.ADVANCE

    failing = gates.evaluate_all(
        hard,
        [
            {"gate_id": "HG1", "candidate_value": 4},
            {"gate_id": "HG2", "candidate_value": "absent"},
        ],
    )
    assert gates.gold_decision(failing) == gates.DO_NOT_ADVANCE


def test_missing_candidate_value_raises():
    with pytest.raises(gates.GateError):
        gates.evaluate_all(
            [numeric_gate(), categorical_gate()],
            [{"gate_id": "HG1", "candidate_value": 4}],
        )


def test_value_for_undeclared_gate_raises():
    with pytest.raises(gates.GateError):
        gates.evaluate_all(
            [numeric_gate()],
            [
                {"gate_id": "HG1", "candidate_value": 4},
                {"gate_id": "HG9", "candidate_value": 1},
            ],
        )


def test_minimum_margin_ignores_categorical_gates():
    results = gates.evaluate_all(
        [numeric_gate(), categorical_gate()],
        [
            {"gate_id": "HG1", "candidate_value": 1},
            {"gate_id": "HG2", "candidate_value": "demonstrated"},
        ],
    )
    assert gates.minimum_gate_margin(results) == -2.0
