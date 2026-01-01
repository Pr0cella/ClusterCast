from __future__ import annotations

import os
import json
from typing import List, Tuple


def validate_output_structure(output_dir: str) -> Tuple[bool, List[str]]:
    """Assert four top-level report files exist in split mode and assets folder is present.
    Returns (ok, messages).
    """
    expected = {'predictions.md', 'predictions.json', 'calibration.md', 'calibration.json'}
    present = set(fn for fn in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, fn)))
    messages: List[str] = []
    ok = True
    missing = sorted(list(expected - present))
    if missing:
        ok = False
        messages.append(f"Missing expected files: {', '.join(missing)}")
    if not os.path.isdir(os.path.join(output_dir, 'calibration_assets')):
        ok = False
        messages.append("Missing calibration_assets directory")
    return ok, messages


def validate_links(calibration_md_path: str, base_dir: str) -> Tuple[bool, List[str]]:
    """Ensure all markdown image and CSV links in calibration.md resolve to existing files."""
    import re
    messages: List[str] = []
    ok = True
    if not os.path.isfile(calibration_md_path):
        return False, [f"File not found: {calibration_md_path}"]
    with open(calibration_md_path, 'r', encoding='utf-8') as f:
        md = f.read()
    # images ![...](path)
    img_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    # inline links [...](path)
    href_links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", md)
    for rel in set(img_links + href_links):
        # skip anchors and external
        if rel.startswith('#') or '://' in rel:
            continue
        path = os.path.normpath(os.path.join(base_dir, rel))
        if not os.path.exists(path):
            ok = False
            messages.append(f"Broken link: {rel} -> {path}")
    return ok, messages


def compare_json_reports(curr_path: str, prev_path: str) -> Tuple[bool, List[str]]:
    """Compare current vs backup predictions/calibration JSON for shape and key presence."""
    messages: List[str] = []
    ok = True
    if not (os.path.isfile(curr_path) and os.path.isfile(prev_path)):
        return False, ["One or both files missing"]
    with open(curr_path, 'r', encoding='utf-8') as f:
        curr = json.load(f)
    with open(prev_path, 'r', encoding='utf-8') as f:
        prev = json.load(f)
    # compare top-level keys
    curr_keys = set(curr.keys())
    prev_keys = set(prev.keys())
    missing_in_curr = prev_keys - curr_keys
    if missing_in_curr:
        ok = False
        messages.append(f"Missing keys in current: {sorted(list(missing_in_curr))}")
    # sanity: predictions size/targets when available
    try:
        curr_models = curr.get('model_forecasts', [])
        prev_models = prev.get('model_forecasts', [])
        if len(curr_models) != len(prev_models):
            messages.append(f"Model count changed: {len(prev_models)} -> {len(curr_models)}")
        # Optional deeper checks could be added here.
    except Exception:
        pass
    return ok, messages
