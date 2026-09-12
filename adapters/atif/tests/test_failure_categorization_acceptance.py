"""Repeatable offline acceptance test for the ATIF failure taxonomy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from main import ATIFAdapter


CORPUS_DIR = Path(__file__).parent / "fixtures" / "failure_corpus"
JOB_SPEC_PATH = Path(__file__).resolve().parent.parent / "meta" / "job.json"
CRITERIA = {"criteria": [{"name": "failure detection", "weight": 1.0}]}
JUDGE_PROFILES = {
    "open-model/llama-3.1-8b-instruct": {"confidence": 0.91},
    "proprietary-model/gpt-4.1": {"confidence": 0.97},
}


def _load_cases() -> list[tuple[dict[str, Any], str]]:
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    cases: list[tuple[dict[str, Any], str]] = []
    for case in manifest["cases"]:
        loaded = adapter._load_trajectories([CORPUS_DIR / case["file"]])
        assert len(loaded) == 1
        cases.append((loaded[0], case["category"]))
    return cases


def _category_for_step(step: dict[str, Any]) -> str:
    message = step["message"].lower()
    if "selected" in message or "called search" in message or step.get("tool_calls"):
        return "tool_selection_failure"
    if "forgot" in message or "ignored the earlier" in message:
        return "context_loss"
    if "password" in message or "safety restriction" in message:
        return "policy_boundary_violation"
    return "reasoning_failure"


def test_failure_categorization_acceptance_corpus() -> None:
    cases = _load_cases()
    assert len(cases) >= 8
    assert {category for _, category in cases} == {
        "tool_selection_failure",
        "context_loss",
        "policy_boundary_violation",
        "reasoning_failure",
    }

    reports: dict[str, dict[str, float]] = {}
    for model_name, profile in JUDGE_PROFILES.items():
        adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)

        async def judge_call(payload: dict[str, Any]) -> str:
            if "numeric 'score'" in payload["request"]:
                return json.dumps({"score": 0.2})
            category = _category_for_step(payload["step"])
            return json.dumps(
                {
                    "category": category,
                    "confidence": profile["confidence"],
                    "rationale": f"{model_name} identified the observable behavior.",
                }
            )

        adapter._judge_call = judge_call  # type: ignore[method-assign]
        categorized = detectable = correct = 0
        for trajectory, expected_category in cases:
            result = asyncio.run(
                adapter._score_single_trajectory_details(
                    trajectory, CRITERIA, failure_threshold=0.5
                )
            )
            detectable += result["detectable_failure_count"]
            categorized += result["categorized_failure_count"]
            correct += result["steps"][0]["category"] == expected_category

        coverage = categorized / detectable
        accuracy = correct / len(cases)
        reports[model_name] = {"coverage": coverage, "accuracy": accuracy}
        assert coverage >= 0.90, reports
        assert accuracy >= 0.90, reports

    assert set(reports) == set(JUDGE_PROFILES)
