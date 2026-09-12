from __future__ import annotations

import json
from pathlib import Path

import respx
from evalhub.adapter import JobCallbacks
from httpx import Response

from main import ATIFAdapter


JOB_SPEC_PATH = Path(__file__).resolve().parent.parent / "meta" / "job.json"


class FakeCallbacks(JobCallbacks):
    def report_status(self, update):
        pass

    def create_oci_artifact(self, spec):
        raise AssertionError("ATIF adapter should not create an OCI artifact")

    def report_results(self, results):
        pass


def test_atif_adapter_happy_path(job_spec):
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    callbacks = FakeCallbacks()

    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://localhost:8080/v1/chat/completions").mock(
            side_effect=[
                Response(200, text=json.dumps({"criteria": [{"name": "quality", "weight": 1.0}]})),
                Response(200, text=json.dumps({"score": 0.8})),
                Response(200, text=json.dumps({"score": 0.6})),
            ]
        )

        result = adapter.run_benchmark_job(job_spec, callbacks)

    assert result.overall_score == 0.7
    assert len(result.results) == 2
    assert "atif_trajectories" in result.evaluation_metadata


def test_discover_directory(tmp_path: Path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "nested" / "b.json"
    f2.parent.mkdir(parents=True)
    f1.write_text("{}")
    f2.write_text("{}")

    files = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)._discover_atif_files(str(tmp_path))
    assert len(files) == 2


def test_judge_429_retry(job_spec):
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    callbacks = FakeCallbacks()

    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://localhost:8080/v1/chat/completions").mock(
            side_effect=[
                Response(429, text="rate limited"),
                Response(200, text=json.dumps({"criteria": [{"name": "quality", "weight": 1.0}]})),
                Response(200, text=json.dumps({"score": 0.9})),
                Response(200, text=json.dumps({"score": 0.9})),
            ]
        )

        result = adapter.run_benchmark_job(job_spec, callbacks)

    assert result.overall_score == 0.9
