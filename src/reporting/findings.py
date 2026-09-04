"""Safe finding sentences (SIH26038 Phase 9). Every sentence pattern uses
screening/evidence language; the banned-phrase list in config guards this."""

GRADE_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


def grade_sentence(grade, labels=None):
    labels = labels or GRADE_LABELS
    if grade not in (0, 1, 2, 3, 4):
        return "DR grade was not assessed for this image."
    return (f"The model predicted {labels.get(grade, '?')} diabetic-retinopathy "
            f"severity (Grade {grade}).")


def lesion_sentence(ltype, status, count):
    name = ltype.replace("_", " ")
    if status == "DETECTED" and count:
        return (f"{name.capitalize()} evidence: DETECTED, {count} candidate "
                f"regions detected by the heuristic module.")
    if status == "LOW_CONFIDENCE":
        return f"{name.capitalize()} evidence: LOW_CONFIDENCE, review suggested."
    return f"No {name} evidence detected by this heuristic module. This does NOT mean the patient definitely has no {name}."


def landmark_sentence(name, status, confidence=None):
    if status in ("DETECTED", "LOW_CONFIDENCE"):
        extra = f" (localization confidence {confidence:.2f})" \
            if confidence is not None else ""
        return f"{name.capitalize()}: {status.lower()}{extra}."
    return f"{name.capitalize()} localization unavailable."


def confidence_sentence(value):
    return (f"Calibrated model confidence: {value:.4f}. "
            f"Calibrated confidence is not clinical certainty.")


def referable_sentence(score, threshold):
    state = "REFERABLE" if score is not None and score >= threshold else "NON_REFERABLE"
    s = f"{score:.4f}" if score is not None else "unavailable"
    return (f"Referable probability: {s} (decision threshold {threshold}). "
            f"Screening classification: {state}.")
