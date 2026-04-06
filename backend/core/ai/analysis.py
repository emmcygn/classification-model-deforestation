"""AI-powered policy brief generation using GPT-4o-mini."""

import os
import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

try:
    from core.data.fetch_philippines import PROVENANCE
except ImportError:
    PROVENANCE = {}


def generate_policy_brief(region_data: dict) -> dict:
    """Generate a structured policy brief for a region."""
    brief = {
        "site_overview": {
            "name": region_data.get("region_name", "Selected Region"),
            "total_cells": region_data["stats"]["total_cells"],
            "area_km2": round(region_data["stats"]["total_cells"] * 5.5 * 5.5, 0),
            "bounds": region_data["stats"]["bounds"],
        },
        "risk_assessment": region_data["risk_distribution"],
        "hotspots": region_data.get("hotspots", []),
        "top_drivers": region_data["top_features"],
        "notable_findings": region_data["notable_points"],
        "data_provenance": [
            {"feature": k, **v} for k, v in PROVENANCE.items()
        ],
        "model_run_id": region_data.get("run_id", ""),
    }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        brief["executive_summary"] = (
            "AI narrative unavailable — set OPENAI_API_KEY for live generation. "
            "All data sections above are generated deterministically from model outputs."
        )
        brief["recommendations"] = [
            "Configure OPENAI_API_KEY to enable AI-generated policy recommendations.",
        ]
        return brief

    client = OpenAI(api_key=api_key)

    prompt = f"""You are a deforestation policy analyst writing a brief for Philippine government officials (DENR/PENRO/CENRO). Given the following structured data, write:

1. An executive summary (2-3 sentences, cite specific numbers)
2. Policy recommendations (3-5 actionable bullets referencing Philippine regulations like NIPAS Act, EO 23 logging moratorium, or DENR administrative orders)

Be specific, cite numbers from the data, and frame recommendations for Philippine regulatory context.

Region: {brief['site_overview']['name']}
Total cells: {brief['site_overview']['total_cells']} (~{brief['site_overview']['area_km2']:.0f} km²)
Risk distribution: {json.dumps(brief['risk_assessment'])}
Top risk drivers: {json.dumps(brief['top_drivers'][:5])}
Notable findings: {json.dumps(brief['notable_findings'][:5])}
Hotspots: {json.dumps(brief['hotspots'][:3])}

Respond in JSON format:
{{"executive_summary": "...", "recommendations": ["...", "..."]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        ai_output = json.loads(response.choices[0].message.content)
        brief["executive_summary"] = ai_output.get("executive_summary", "")
        brief["recommendations"] = ai_output.get("recommendations", [])
    except Exception as e:
        logger.warning("Policy brief AI generation failed: %s", e)
        brief["executive_summary"] = "AI generation unavailable for this request."
        brief["recommendations"] = []

    return brief
