from __future__ import annotations

import json
from pathlib import Path

import pytest
import respx
from evalhub.adapter import JobCallbacks
from httpx import Response

from main import ATIFAdapter, ATIFLoadError, JudgeResponseError


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
    assert len(result.results) == 3
    assert "atif_trajectories" in result.evaluation_metadata


def test_training_eligibility_marks_trajectories_and_manifest(job_spec):
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    callbacks = FakeCallbacks()
    job_spec = job_spec.model_copy(
        update={
            "parameters": {
                **(job_spec.parameters or {}),
                "training_threshold": 0.65,
            }
        }
    )

    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://localhost:8080/v1/chat/completions").mock(
            side_effect=[
                Response(200, text=json.dumps({"criteria": [{"name": "quality"}]})),
                Response(200, text=json.dumps({"score": 0.8})),
                Response(200, text=json.dumps({"score": 0.6})),
            ]
        )

        result = adapter.run_benchmark_job(job_spec, callbacks)

    trajectory = result.evaluation_metadata["atif_trajectories"][0]
    assert trajectory["training_eligible"] is True
    assert trajectory["source_path"].endswith("trajectory.json")
    assert result.evaluation_metadata["atif_training_threshold"] == 0.65
    assert result.evaluation_metadata["atif_training_manifest"] == [
        trajectory["source_path"]
    ]


@pytest.mark.parametrize("training_threshold", [-0.1, 1.1])
def test_invalid_training_threshold_fails_fast(job_spec, training_threshold):
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    callbacks = FakeCallbacks()
    job_spec = job_spec.model_copy(
        update={
            "parameters": {
                **(job_spec.parameters or {}),
                "training_threshold": training_threshold,
            }
        }
    )

    with pytest.raises(ValueError, match="training_threshold"):
        adapter.run_benchmark_job(job_spec, callbacks)


def test_failure_categorization_for_low_scoring_step(job_spec):
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    callbacks = FakeCallbacks()

    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://localhost:8080/v1/chat/completions").mock(
            side_effect=[
                Response(200, text=json.dumps({"criteria": [{"name": "quality"}]})),
                Response(200, text=json.dumps({"score": 0.2})),
                Response(
                    200,
                    text=json.dumps(
                        {
                            "category": "reasoning_failure",
                            "confidence": 0.9,
                            "rationale": "The action contradicts the preceding observation.",
                        }
                    ),
                ),
                Response(200, text=json.dumps({"score": 0.8})),
            ]
        )

        result = adapter.run_benchmark_job(job_spec, callbacks)

    trajectory = result.evaluation_metadata["atif_trajectories"][0]
    assert trajectory["detectable_failure_count"] == 1
    assert trajectory["categorized_failure_count"] == 1
    assert trajectory["steps"][0]["category"] == "reasoning_failure"
    assert result.evaluation_metadata["atif_failure_categorization_rate"] == 1.0


def test_invalid_failure_category_is_uncategorized_with_raw_response(job_spec):
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    callbacks = FakeCallbacks()

    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://localhost:8080/v1/chat/completions").mock(
            side_effect=[
                Response(200, text=json.dumps({"criteria": [{"name": "quality"}]})),
                Response(200, text=json.dumps({"score": 0.2})),
                Response(200, text="not valid category JSON"),
                Response(200, text=json.dumps({"score": 0.8})),
            ]
        )

        result = adapter.run_benchmark_job(job_spec, callbacks)

    step = result.evaluation_metadata["atif_trajectories"][0]["steps"][0]
    assert step["category"] == "uncategorized"
    assert step["raw_judge_response"] == "not valid category JSON"
    assert result.evaluation_metadata["atif_failure_categorization_rate"] == 0.0


@pytest.mark.parametrize(
    "score_response",
    [
        "not valid JSON",
        json.dumps({}),
        json.dumps({"score": "not-a-number"}),
        json.dumps({"score": 1.1}),
    ],
)
def test_invalid_score_response_fails_closed(job_spec, score_response):
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    callbacks = FakeCallbacks()

    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://localhost:8080/v1/chat/completions").mock(
            side_effect=[
                Response(200, text=json.dumps({"criteria": [{"name": "quality"}]})),
                Response(200, text=score_response),
            ]
        )

        with pytest.raises(JudgeResponseError, match="judge"):
            adapter.run_benchmark_job(job_spec, callbacks)


def test_zero_score_is_preserved_as_a_valid_score(job_spec):
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    callbacks = FakeCallbacks()

    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://localhost:8080/v1/chat/completions").mock(
            side_effect=[
                Response(200, text=json.dumps({"criteria": [{"name": "quality"}]})),
                Response(200, text=json.dumps({"score": 0.0})),
                Response(200, text=json.dumps({"category": "none", "confidence": 1.0})),
                Response(200, text=json.dumps({"score": 0.0})),
                Response(200, text=json.dumps({"category": "none", "confidence": 1.0})),
            ]
        )

        result = adapter.run_benchmark_job(job_spec, callbacks)

    assert result.overall_score == 0.0
    assert result.evaluation_metadata["atif_detectable_failure_count"] == 2


def test_discover_directory(tmp_path: Path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "nested" / "b.json"
    f2.parent.mkdir(parents=True)
    f1.write_text("{}")
    f2.write_text("{}")

    files = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)._discover_atif_files(str(tmp_path))
    assert len(files) == 2


def _valid_trajectory() -> dict:
    return json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "trajectory.json").read_text()
    )


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_load_valid_trajectory_uses_sdk_model(tmp_path: Path):
    trajectory_file = _write_json(tmp_path / "valid.json", _valid_trajectory())
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)

    loaded = adapter._load_trajectories([trajectory_file])

    assert len(loaded) == 1
    assert loaded[0]["schema_version"] == "ATIF-v1.8"
    assert loaded[0]["trajectory_id"] == "t1"


def test_supported_schema_versions_come_from_sdk_model():
    assert ATIFAdapter._supported_schema_versions() == frozenset(
        f"ATIF-v1.{version}" for version in range(9)
    )


def test_load_rejects_malformed_json(tmp_path: Path):
    trajectory_file = tmp_path / "malformed.json"
    trajectory_file.write_text("{not-json")
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)

    with pytest.raises(ATIFLoadError, match="invalid ATIF trajectory"):
        adapter._load_trajectories([trajectory_file])


def test_load_rejects_invalid_trajectory_using_sdk_validation(tmp_path: Path):
    trajectory = _valid_trajectory()
    trajectory["steps"][1]["step_id"] = 3
    trajectory_file = _write_json(tmp_path / "invalid-step.json", trajectory)
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)

    with pytest.raises(ATIFLoadError, match="expected 2"):
        adapter._load_trajectories([trajectory_file])


def test_load_rejects_unsupported_schema_version(tmp_path: Path):
    trajectory = _valid_trajectory()
    trajectory["schema_version"] = "ATIF-v9.9"
    trajectory_file = _write_json(tmp_path / "unsupported-version.json", trajectory)
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)

    with pytest.raises(ATIFLoadError, match="unsupported schema_version"):
        adapter._load_trajectories([trajectory_file])


def test_load_rejects_empty_collection(tmp_path: Path):
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)

    with pytest.raises(ATIFLoadError, match="No ATIF JSON"):
        adapter._load_trajectories([])


def test_load_rejects_duplicate_trajectory_ids(tmp_path: Path):
    trajectory = _valid_trajectory()
    first = _write_json(tmp_path / "first.json", trajectory)
    second = _write_json(tmp_path / "second.json", trajectory)
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)

    with pytest.raises(ATIFLoadError, match="duplicate trajectory_id"):
        adapter._load_trajectories([first, second])


def test_load_rejects_file_and_step_limits(tmp_path: Path):
    trajectory_file = _write_json(tmp_path / "valid.json", _valid_trajectory())
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)

    with pytest.raises(ATIFLoadError, match="exceeds maximum"):
        adapter._load_trajectories([trajectory_file], max_file_bytes=1)

    with pytest.raises(ATIFLoadError, match="maximum is 1"):
        adapter._load_trajectories([trajectory_file], max_steps_per_trajectory=1)


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
