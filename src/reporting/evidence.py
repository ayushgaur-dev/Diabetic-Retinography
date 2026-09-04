"""Evidence tables (SIH26038 Phase 9). Rows only from supplied structs."""

LESION_ORDER = ("microaneurysm", "hemorrhage", "hard_exudate", "soft_exudate")


def lesion_rows(lesions):
    rows = []
    for lt in LESION_ORDER:
        d = (lesions or {}).get(lt, {})
        rows.append({
            "lesion_type": lt,
            "status": d.get("status", "UNKNOWN"),
            "candidate_count": d.get("candidate_count", 0),
            "evidence_area": d.get("total_evidence_area"),
            "confidence": d.get("confidence"),
        })
    return rows


def anatomy_rows(optic_disc, fovea, vessels):
    od = optic_disc or {}
    fo = fovea or {}
    return [
        {"structure": "optic_disc",
         "status": od.get("status", "UNKNOWN"),
         "region": od.get("center") or od.get("bounding_box"),
         "confidence": od.get("confidence")},
        {"structure": "fovea",
         "status": fo.get("status", "UNKNOWN"),
         "region": fo.get("center_x_y"),
         "confidence": fo.get("confidence")},
        {"structure": "vessels",
         "status": "available" if vessels else "unavailable",
         "region": None, "confidence": None},
    ]
