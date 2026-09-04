"""Markdown renderer: single rendering of the authoritative dict."""

GRADE_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


def render_markdown(report):
    s = report["sections"]
    L = ["# AI-Assisted Retinal Screening Report", ""]
    L.append(f"Image: `{s.get('image_id')}`")
    L.append(f"Report status: {report.get('status')}")
    L.append("")
    sm = s.get("summary", {})
    L += ["## Screening Summary", "",
          f"Image quality: **{sm.get('image_quality')}**",
          f"Model prediction: **{sm.get('model_prediction')}**",
          f"Calibrated confidence: **{sm.get('calibrated_confidence')}**",
          f"Referable probability: **{sm.get('referable_probability')}**",
          f"Workflow recommendation: **{sm.get('workflow_recommendation')}**",
          f"Priority: **{sm.get('priority')}**", ""]
    q = s.get("quality", {})
    L += ["## Image Quality", "", f"Status: **{q.get('status')}**"]
    comps = q.get("components", {}) or {}
    for k, v in comps.items():
        L.append(f"- {k}: {v}")
    for r in q.get("recapture_feedback", []) or []:
        L.append(f"- Recapture: {r}")
    if q.get("notice"):
        L += ["", q["notice"]]
    L += [""]
    e = s.get("enhancement", {})
    L += ["## Enhancement", "", f"Status: **{e.get('status')}**"]
    if e.get("operations"):
        L.append(f"Operations: {', '.join(e['operations'])}")
    if e.get("before") or e.get("after"):
        L.append(f"Quality: {e.get('before')} → {e.get('after')}")
    if e.get("note"):
        L.append(e["note"])
    L += [""]
    g = s.get("grading", {})
    L += ["## DR Grading", ""]
    if not g.get("assessed"):
        L.append(g.get("notice", "Not assessed."))
    else:
        for i, (rp, cp) in enumerate(zip(g.get("raw_probabilities", []),
                                         g.get("calibrated_probabilities", []))):
            L.append(f"- Grade {i} ({GRADE_LABELS.get(i, '?')}): raw {rp:.4f} / "
                     f"calibrated {cp:.4f}")
        L.append(f"Predicted grade: **{g['predicted_grade']} — "
                 f"{g.get('grade_label')}**")
        if g.get("confidence_sentence"):
            L.append(g["confidence_sentence"])
    L += [""]
    rf = s.get("referable", {})
    L += ["## Referable Screening", "",
          f"Referable probability: **{rf.get('score')}**",
          f"Decision threshold: **{rf.get('threshold')}**",
          f"Screening classification: **{rf.get('classification')}**",
          rf.get("definition", ""), rf.get("threshold_note", ""), ""]
    ls = s.get("lesions", {})
    L += ["## Lesion Evidence", ""]
    for row in ls.get("lesions", []) or []:
        L.append(f"### {row['lesion_type'].replace('_', ' ').title()}")
        L.append(f"Status: **{row['status']}** — candidates: {row['candidate_count']}")
        for c in row.get("top_candidates", [])[:5]:
            L.append(f"- bbox {c.get('bbox')}, score {c.get('score')}")
    if ls.get("limitation"):
        L += ["", "*" + ls["limitation"] + "*"]
    L += [""]
    an = s.get("anatomy", {})
    L += ["## Anatomical Evidence", ""]
    for row in an.get("structures", []) or []:
        L.append(f"- {row['structure']}: {row['status']}")
    for sent in an.get("sentences", []) or []:
        L.append(sent)
    L += [""]
    ex = s.get("explainability", {})
    L += ["## Explainability", "",
          f"Consistency: **{ex.get('consistency')}**",
          ex.get("consistency_reason", "")]
    for lt, v in (ex.get("lesion_agreement", {}) or {}).items():
        L.append(f"- {lt}: spatial agreement {v}")
    if ex.get("figure"):
        L.append(f"Explainability figure: {ex['figure']}")
    L.append(ex.get("note", ""))
    L += [""]
    tr = s.get("triage", {})
    L += ["## Workflow Recommendation", "",
          f"Recommendation: **{tr.get('decision')}**",
          f"Priority: **{tr.get('priority')}**",
          tr.get("workflow", "")]
    L.append("Reasons:")
    for code, text in (tr.get("reason_translations") or {}).items():
        L.append(f"- {code}: {text}")
    flags = tr.get("safety_flags", {}) or {}
    raised = [k for k, v in flags.items() if v]
    L.append(f"Safety flags raised: {', '.join(raised) if raised else 'none'}")
    if tr.get("explanation"):
        L += ["", tr["explanation"]]
    L += [""]
    L += ["## Limitations", ""]
    for lim in s.get("limitations", []) or []:
        L.append(f"- {lim}")
    if s.get("warnings"):
        L += ["", "## Warnings", ""]
        for w in s["warnings"]:
            L.append(f"- {w}")
    L += [""]
    pv = s.get("provenance", {}) if isinstance(s.get("provenance"), dict) else {}
    L += ["## Provenance", ""]
    for k, v in pv.items():
        L.append(f"- {k}: {v}")
    L.append("")
    return "\n".join(L)
