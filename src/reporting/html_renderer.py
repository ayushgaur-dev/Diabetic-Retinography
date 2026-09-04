"""HTML renderer: standalone offline report (inline CSS, no CDN).
Optional explainability figure embedded as base64 when a path is supplied."""

import base64
import html
from pathlib import Path


def _embed_figure(path):
    try:
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" style="max-width:100%"/>'
    except (FileNotFoundError, OSError, ValueError):
        return ""


def render_html(report, figure_path=None):
    s = report["sections"]
    sm = s.get("summary", {})
    tr = s.get("triage", {})
    g = s.get("grading", {})
    rf = s.get("referable", {})
    ls = s.get("lesions", {})
    an = s.get("anatomy", {})
    ex = s.get("explainability", {})
    q = s.get("quality", {})
    e = lambda v: html.escape(str(v))
    rows = "".join(
        f"<tr><td>{e(r['lesion_type'])}</td><td>{e(r['status'])}</td>"
        f"<td>{e(r['candidate_count'])}</td></tr>"
        for r in ls.get("lesions", []) or [])
    anat = "".join(
        f"<tr><td>{e(r['structure'])}</td><td>{e(r['status'])}</td></tr>"
        for r in an.get("structures", []) or [])
    probs = "".join(
        f"<tr><td>Grade {i}</td><td>{rp}</td><td>{cp}</td></tr>"
        for i, (rp, cp) in enumerate(zip(g.get("raw_probabilities", []),
                                         g.get("calibrated_probabilities", []))))
    reasons = "".join(f"<li><b>{e(c)}</b>: {e(t)}</li>"
                      for c, t in (tr.get("reason_translations") or {}).items())
    flags = tr.get("safety_flags", {}) or {}
    raised = ", ".join(k for k, v in flags.items() if v) or "none"
    lims = "".join(f"<li>{e(x)}</li>" for x in s.get("limitations", []) or [])
    fig = _embed_figure(figure_path or (ex.get("figure") or "")) if (
        figure_path or ex.get("figure")) else "<p>Figure unavailable.</p>"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>AI-Assisted Retinal Screening Report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}}
.card{{border:1px solid #999;border-radius:8px;padding:12px;margin:12px 0}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:4px 8px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:10px;background:#eee}}</style>
</head><body>
<h1>AI-Assisted Retinal Screening Report</h1>
<p>Image: <code>{e(s.get('image_id'))}</code> | Status: {e(report.get('status'))}</p>
<div class="card"><h2>Screening Summary</h2>
<p>Quality: <span class="badge">{e(sm.get('image_quality'))}</span>
Prediction: <b>{e(sm.get('model_prediction'))}</b></p>
<p>Calibrated confidence: <b>{e(sm.get('calibrated_confidence'))}</b> |
Referable probability: <b>{e(sm.get('referable_probability'))}</b></p>
<p>Workflow: <b>{e(sm.get('workflow_recommendation'))}</b> |
Priority: <b>{e(sm.get('priority'))}</b></p></div>
<div class="card"><h2>Image Quality</h2><p>Status: {e(q.get('status'))}</p></div>
<div class="card"><h2>DR Grading</h2>
<p>{e(g.get('sentence', g.get('notice', '')))}</p>
<table><tr><th>Grade</th><th>Raw</th><th>Calibrated</th></tr>{probs}</table></div>
<div class="card"><h2>Referable Screening</h2>
<p>Score {e(rf.get('score'))} vs threshold {e(rf.get('threshold'))}:
<b>{e(rf.get('classification'))}</b></p></div>
<div class="card"><h2>Lesion Evidence</h2>
<table><tr><th>Type</th><th>Status</th><th>Candidates</th></tr>{rows}</table>
<p><i>{e(ls.get('limitation', ''))}</i></p></div>
<div class="card"><h2>Anatomical Evidence</h2>
<table><tr><th>Structure</th><th>Status</th></tr>{anat}</table></div>
<div class="card"><h2>Explainability</h2>
<p>Consistency: <b>{e(ex.get('consistency'))}</b></p>{fig}</div>
<div class="card"><h2>Workflow Recommendation</h2>
<p><b>{e(tr.get('decision'))}</b> ({e(tr.get('priority'))})</p>
<p>{e(tr.get('workflow'))}</p><ul>{reasons}</ul>
<p>Safety flags raised: {e(raised)}</p></div>
<div class="card"><h2>Limitations</h2><ul>{lims}</ul></div>
</body></html>"""
