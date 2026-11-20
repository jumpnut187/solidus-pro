from __future__ import annotations

import html
from typing import Dict

from core.result_help import result_help_text

RESULT_TYPE_MODES: Dict[str, str] = {
    "Shrink": "shrinkage",
    "Microporosity": "porosity",
    "Feed Modulus": "feed_modulus",
    "Hot Spots": "hotspot",
    "Niyama Criterion": "niyama",
    "Hot Tear Risk": "hot_tear",
    "Sand Burn-In": "sand_burn_in",
    "Gas Generation": "gas_generation",
    "Gas Entrapment": "gas_entrapment",
    "Gas Porosity Risk": "gas_porosity",
    "Fraction Liquid": "fraction_liquid",
    "Fill Time": "fill_time",
    "Fill Animation": "fill_animation",
    "Feeding Quality": "feeding_quality",
    "Thermal Stress": "thermal_stress",
    "Distortion": "distortion",
    "Crack Susceptibility": "crack_susceptibility",
    "SDAS (Microstructure)": "sdas",
    "Predicted UTS": "predicted_uts",
}

LEGACY_RESULT_TYPE_LABELS: Dict[str, str] = {
    "Shrinkage/Porosity": "Shrink",
}

DEFAULT_RESULT_TYPE = "Shrink"

def canonical_result_type(label: str) -> str:
    return LEGACY_RESULT_TYPE_LABELS.get(label, label)

def result_type_mode(label: str) -> str:
    return RESULT_TYPE_MODES.get(canonical_result_type(label), "shrinkage")

def result_type_help(label: str) -> str:
    mode = RESULT_TYPE_MODES.get(canonical_result_type(label))
    return result_help_text(mode) if mode is not None else ""

def result_type_help_html(label: str, *, with_title: bool = False) -> str:
    text = result_type_help(label)
    if not text:
        return ""
    body = html.escape(text)
    if with_title:
        return f"<p><b>{html.escape(canonical_result_type(label))}</b><br>{body}</p>"
    return f"<p>{body}</p>"
