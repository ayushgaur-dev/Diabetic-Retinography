import sys
from pathlib import Path

# Make `src` importable regardless of how/where pytest is invoked from,
# without requiring __init__.py files or a package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.rules.triage import triage_decision, BASE_RULES, CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 1. Base action/interval per grade (BASE_RULES)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grade", sorted(BASE_RULES.keys()))
def test_base_rules_per_grade(grade):
    # confidence=0.99 stays clear of both the abstention gate (0.60) and
    # the intermediate-grade safety net (0.75), isolating BASE_RULES.
    decision = triage_decision(grade, confidence=0.99)
    expected = BASE_RULES[grade]
    assert decision['action'] == expected['action']
    assert decision['interval_months'] == expected['interval_months']
    assert decision['requires_human_review'] is False


# ---------------------------------------------------------------------------
# 2. Abstention below CONFIDENCE_THRESHOLD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grade", range(5))
def test_abstains_below_confidence_threshold(grade):
    decision = triage_decision(grade, confidence=0.50)
    assert decision['requires_human_review'] is True
    assert decision['action'] == 'abstain_low_confidence'
    assert decision['interval_months'] is None
    assert any('below threshold' in note for note in decision['notes'])


# ---------------------------------------------------------------------------
# 3. Intermediate-grade safety net (grades 1-3, confidence 0.60-0.75)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grade", [1, 2, 3])
def test_intermediate_grade_flagged_between_060_and_075(grade):
    decision = triage_decision(grade, confidence=0.65)
    assert decision['requires_human_review'] is True
    assert any('lower reliability' in note for note in decision['notes'])


@pytest.mark.parametrize("grade", [1, 2, 3])
def test_intermediate_grade_not_flagged_at_or_above_075(grade):
    decision = triage_decision(grade, confidence=0.75)
    assert decision['requires_human_review'] is False
    assert decision['notes'] == []


@pytest.mark.parametrize("grade", [0, 4])
def test_safety_net_does_not_apply_outside_grades_1_to_3(grade):
    # Same confidence band that would trigger the safety net for grades 1-3.
    decision = triage_decision(grade, confidence=0.65)
    assert decision['requires_human_review'] is False
    assert decision['notes'] == []


# ---------------------------------------------------------------------------
# 4. Patient modifiers, individually
# ---------------------------------------------------------------------------

def test_high_hba1c_halves_interval():
    decision = triage_decision(0, confidence=0.99, patient_params={'hba1c': 9.5})
    assert decision['interval_months'] == 6  # 12 // 2
    assert any('glycaemic control' in note for note in decision['notes'])


def test_hba1c_below_threshold_no_effect():
    decision = triage_decision(0, confidence=0.99, patient_params={'hba1c': 8.9})
    assert decision['interval_months'] == 12
    assert decision['notes'] == []


def test_hba1c_does_not_shorten_interval_already_at_or_below_3():
    # Grade 3 base interval is already 1 month; the modifier only fires
    # when interval_months > 3, so it must stay untouched.
    decision = triage_decision(3, confidence=0.99, patient_params={'hba1c': 9.5})
    assert decision['interval_months'] == 1
    assert decision['notes'] == []


def test_long_diabetes_duration_shortens_interval():
    decision = triage_decision(1, confidence=0.99, patient_params={'diabetes_years': 20})
    assert decision['interval_months'] == 3  # 6 - 3
    assert any('duration' in note for note in decision['notes'])


def test_long_diabetes_duration_requires_nonzero_grade():
    # Grade 0 is explicitly excluded by the `predicted_grade > 0` guard.
    decision = triage_decision(0, confidence=0.99, patient_params={'diabetes_years': 20})
    assert decision['interval_months'] == 12
    assert decision['notes'] == []


def test_diabetes_years_below_threshold_no_effect():
    decision = triage_decision(1, confidence=0.99, patient_params={'diabetes_years': 14})
    assert decision['interval_months'] == 6
    assert decision['notes'] == []


def test_high_myopia_flags_note_without_changing_interval_or_action():
    decision = triage_decision(0, confidence=0.99, patient_params={'spherical_refraction': -6.5})
    assert decision['interval_months'] == 12
    assert decision['action'] == BASE_RULES[0]['action']
    assert any('myopia' in note for note in decision['notes'])


def test_myopia_at_boundary_exactly_minus_6_triggers():
    decision = triage_decision(0, confidence=0.99, patient_params={'spherical_refraction': -6.0})
    assert any('myopia' in note for note in decision['notes'])


def test_myopia_below_threshold_no_note():
    decision = triage_decision(0, confidence=0.99, patient_params={'spherical_refraction': -5.9})
    assert decision['notes'] == []


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

def test_confidence_exactly_at_threshold_does_not_abstain():
    decision = triage_decision(0, confidence=0.60)
    assert decision['action'] != 'abstain_low_confidence'
    assert decision['interval_months'] == 12
    assert decision['requires_human_review'] is False


def test_confidence_exactly_at_threshold_intermediate_grade_still_flagged():
    # 0.60 clears abstention (not < 0.60) but is still < 0.75, so the
    # intermediate-grade safety net still applies for grades 1-3.
    decision = triage_decision(2, confidence=0.60)
    assert decision['action'] != 'abstain_low_confidence'
    assert decision['requires_human_review'] is True


def test_grade_boundary_0():
    decision = triage_decision(0, confidence=0.99)
    assert decision['action'] == 'virtual_followup'
    assert decision['interval_months'] == 12
    assert decision['requires_human_review'] is False


def test_grade_boundary_4():
    decision = triage_decision(4, confidence=0.99)
    assert decision['action'] == 'urgent_referral'
    assert decision['interval_months'] == 1
    assert decision['requires_human_review'] is False
