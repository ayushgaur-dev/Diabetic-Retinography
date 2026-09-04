"""JSON renderer: authoritative serialization (sorted keys for byte-equivalence)."""

import json


def render_json(report_dict):
    return json.dumps(report_dict, indent=2, sort_keys=True)
