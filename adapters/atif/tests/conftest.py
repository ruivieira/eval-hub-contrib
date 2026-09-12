from __future__ import annotations

import json
from pathlib import Path

import pytest
from evalhub.adapter import JobSpec


@pytest.fixture
def job_spec() -> JobSpec:
    base = Path(__file__).resolve().parent.parent
    payload = json.loads((base / "meta" / "job.json").read_text())
    return JobSpec.model_validate(payload)
