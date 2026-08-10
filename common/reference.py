"""Reference robustness pools: load the reference order from XML and score agreement."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Literal, Optional, Sequence, Tuple


def load_reference_xml(xml_path: str) -> List[Dict[str, object]]:
    root = ET.parse(xml_path).getroot()
    entries: List[Dict[str, object]] = []
    for node in root.findall("model"):
        entries.append(
            {
                "model_id": node.attrib["id"],
                "clean_acc": float(node.attrib.get("clean_acc", "0")),
                "robust_acc": float(node.attrib.get("robust_acc", "0")),
                "architecture": node.attrib.get("architecture", node.attrib["id"]),
            }
        )
    return entries


def models_from_reference_xml(
    reference_entries: Sequence[Dict[str, object]],
) -> Tuple[List[str], List[str], Dict[str, Dict[str, object]]]:
    """All models listed in the reference XML (order preserved)."""
    if not reference_entries:
        raise ValueError("No models in reference XML")
    reference_by_id = {str(e["model_id"]): e for e in reference_entries}
    model_names = [str(e["model_id"]) for e in reference_entries]
    model_labels = [str(e.get("architecture", e["model_id"])) for e in reference_entries]
    return model_names, model_labels, reference_by_id


def assign_reference_ranks(entries: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """Competition ranks 1..N by robust_acc; ties share rank (1,2,2,4,...)."""
    ordered = sorted(
        entries,
        key=lambda e: (-float(e["robust_acc"]), str(e["model_id"])),
    )
    ranks: Dict[str, int] = {}
    i = 0
    n = len(ordered)
    while i < n:
        j = i
        robust_i = float(ordered[i]["robust_acc"])
        while j + 1 < n and float(ordered[j + 1]["robust_acc"]) == robust_i:
            j += 1
        rank = i + 1
        for k in range(i, j + 1):
            ranks[str(ordered[k]["model_id"])] = rank
        i = j + 1
    return ranks


def reference_tied(
    entry_a: Dict[str, object],
    entry_b: Dict[str, object],
) -> bool:
    return float(entry_a["robust_acc"]) == float(entry_b["robust_acc"])


def reference_relation(
    entry_a: Dict[str, object],
    entry_b: Dict[str, object],
) -> Optional[Literal["a_better", "b_better"]]:
    """Order by robust_acc only; None when tied."""
    ra, rb = float(entry_a["robust_acc"]), float(entry_b["robust_acc"])
    if ra == rb:
        return None
    return "a_better" if ra > rb else "b_better"


def assign_score_ranks_with_ties(
    model_ids: Sequence[str],
    scores: Dict[str, float],
) -> Dict[str, int]:
    """
    Competition ranks 1..N by score desc; equal scores share the same rank.

    SGA-score ties count as disagreements whenever the reference order is strict.
    """
    ordered = sorted(model_ids, key=lambda m: (-scores[m], m))
    ranks: Dict[str, int] = {}
    i = 0
    n = len(ordered)
    while i < n:
        j = i
        while j + 1 < n and scores[ordered[j + 1]] == scores[ordered[i]]:
            j += 1
        rank = i + 1
        for k in range(i, j + 1):
            ranks[ordered[k]] = rank
        i = j + 1
    return ranks


def score_pairwise_relation(
    pred_ranks: Dict[str, int],
    a: str,
    b: str,
) -> Optional[Literal["a_better", "b_better"]]:
    ra, rb = pred_ranks[a], pred_ranks[b]
    if ra == rb:
        return None
    return "a_better" if ra < rb else "b_better"


def write_reference_xml(xml_path: str, entries: Sequence[Dict[str, object]]) -> None:
    """Write a reference pool XML."""
    lines = ["<?xml version='1.0' encoding='utf-8'?>", "<reference>"]
    for e in entries:
        mid = str(e["model_id"])
        clean = float(e["clean_acc"])
        robust = float(e["robust_acc"])
        arch = str(e.get("architecture", mid)).replace('"', "&quot;")
        lines.append(
            f'  <model id="{mid}" clean_acc="{clean:.6f}" robust_acc="{robust:.6f}" '
            f'architecture="{arch}" />'
        )
    lines.append("</reference>")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
