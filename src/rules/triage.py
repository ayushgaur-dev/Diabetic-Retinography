"""Triage rule engine.

Pure decision-table function: model output (predicted grade, confidence)
plus optional patient parameters in, a structured triage decision out.
No model dependency — testable without loading the network.

Extracted from the notebook where the base logic was originally
prototyped and validated.
"""

BASE_RULES = {
    0: {'action': 'virtual_followup', 'interval_months': 12},
    1: {'action': 'virtual_followup', 'interval_months': 6},
    2: {'action': 'ophthalmologist_review', 'interval_months': 6},
    3: {'action': 'urgent_referral', 'interval_months': 1},
    4: {'action': 'urgent_referral', 'interval_months': 1},
}

CONFIDENCE_THRESHOLD = 0.60


def triage_decision(predicted_grade, confidence, patient_params=None):
    decision = {
        'predicted_grade': int(predicted_grade),
        'confidence': float(confidence),
        'requires_human_review': False,
        'action': None,
        'interval_months': None,
        'notes': []
    }
    if confidence < CONFIDENCE_THRESHOLD:
        decision['requires_human_review'] = True
        decision['action'] = 'abstain_low_confidence'
        decision['interval_months'] = None
        decision['notes'].append(
            f"Model confidence ({confidence:.2f}) below threshold "
            f"({CONFIDENCE_THRESHOLD}) — deferred to clinician review."
        )
        return decision
    base = BASE_RULES[predicted_grade]
    decision['action'] = base['action']
    decision['interval_months'] = base['interval_months']
    if predicted_grade in [1, 2, 3] and confidence < 0.75:
        decision['requires_human_review'] = True
        decision['notes'].append(
            f"Grade {predicted_grade} predictions have demonstrated lower "
            f"reliability (see Grad-CAM analysis) — flagged for review "
            f"despite passing the base confidence threshold."
        )
    if patient_params:
        decision = _apply_patient_modifiers(decision, patient_params)
    return decision


def _apply_patient_modifiers(decision, params):
    hba1c = params.get('hba1c')
    diabetes_years = params.get('diabetes_years')
    spherical_refraction = params.get('spherical_refraction')

    if hba1c is not None and hba1c >= 9.0:
        if decision['interval_months'] and decision['interval_months'] > 3:
            decision['interval_months'] = max(3, decision['interval_months'] // 2)
            decision['notes'].append(
                f"HbA1c {hba1c}% indicates poor glycaemic control — "
                f"follow-up interval shortened."
            )

    if diabetes_years is not None and diabetes_years >= 15 and decision['predicted_grade'] > 0:
        if decision['interval_months'] and decision['interval_months'] > 3:
            decision['interval_months'] = max(3, decision['interval_months'] - 3)
            decision['notes'].append(
                f"{diabetes_years} years since diabetes diagnosis — "
                f"interval adjusted for long-standing disease duration."
            )

    if spherical_refraction is not None and spherical_refraction <= -6.0:
        decision['notes'].append(
            f"High myopia detected (SE {spherical_refraction} D) — myopic "
            f"maculopathy grading is out of scope for this model; flagged "
            f"for clinician awareness only."
        )

    return decision
