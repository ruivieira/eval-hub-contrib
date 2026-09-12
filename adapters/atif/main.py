#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import httpx
from pydantic import ValidationError
from evalhub.adapter import (
    DefaultCallbacks,
    EvaluationResult,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
)
from evalhub.adapter.auth import resolve_model_credentials
from evalhub.models.atif import Trajectory

logger = logging.getLogger(__name__)


MAX_ATIF_FILE_BYTES = 10 * 1024 * 1024
MAX_ATIF_FILES = 10_000
MAX_STEPS_PER_TRAJECTORY = 500
DEFAULT_FAILURE_THRESHOLD = 0.5
FAILURE_CATEGORIES = frozenset(
    {
        "tool_selection_failure",
        "context_loss",
        "policy_boundary_violation",
        "reasoning_failure",
        "none",
    }
)


class ATIFLoadError(ValueError):
    """Raised when an ATIF input collection cannot be loaded safely."""


class JudgeResponseError(ValueError):
    """Raised when the judge does not return a valid score response."""


class ATIFAdapter(FrameworkAdapter):
    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.monotonic()
        callbacks.report_status(
            JobStatusUpdate(
                phase=JobPhase.INITIALIZING,
                status=JobStatus.RUNNING,
                message=MessageInfo(
                    message="Initializing ATIF adapter", message_code="initializing"
                ),
            )
        )

        params = config.parameters or {}
        concurrency_limit = int(params.get("concurrency_limit", 10))
        trajectory_path = params.get("trajectory_path", "/test_data/trajectory.json")
        max_file_bytes = int(params.get("max_file_bytes", MAX_ATIF_FILE_BYTES))
        max_files = int(params.get("max_trajectory_files", MAX_ATIF_FILES))
        max_steps = int(
            params.get("max_steps_per_trajectory", MAX_STEPS_PER_TRAJECTORY)
        )
        failure_threshold = float(
            params.get("failure_threshold", DEFAULT_FAILURE_THRESHOLD)
        )
        if not 0.0 <= failure_threshold <= 1.0:
            raise ValueError("failure_threshold must be between 0 and 1")
        training_threshold_value = params.get("training_threshold")
        training_threshold = (
            float(training_threshold_value)
            if training_threshold_value is not None
            else None
        )
        if training_threshold is not None and not 0.0 <= training_threshold <= 1.0:
            raise ValueError("training_threshold must be between 0 and 1")

        callbacks.report_status(
            JobStatusUpdate(
                phase=JobPhase.LOADING_DATA,
                status=JobStatus.RUNNING,
                message=MessageInfo(
                    message="Discovering ATIF trajectory files", message_code="loading_data"
                ),
            )
        )
        atif_files = self._discover_atif_files(str(trajectory_path))

        callbacks.report_status(
            JobStatusUpdate(
                phase=JobPhase.RUNNING_EVALUATION,
                status=JobStatus.RUNNING,
                message=MessageInfo(
                    message=f"Scoring {len(atif_files)} ATIF trajectories",
                    message_code="running_evaluation",
                ),
            )
        )

        trajectories = self._load_trajectories(
            atif_files,
            max_file_bytes=max_file_bytes,
            max_files=max_files,
            max_steps_per_trajectory=max_steps,
        )
        scored = asyncio.run(
            self._score_trajectories(
                trajectories, concurrency_limit, failure_threshold=failure_threshold
            )
        )
        eligible_paths: list[str] = []
        for item in scored:
            eligible = (
                None
                if training_threshold is None
                else item["score"] >= training_threshold
            )
            item["training_eligible"] = eligible
            if eligible:
                source_path = item.get("source_path")
                if source_path is not None:
                    eligible_paths.append(source_path)

        avg_score = sum(item["score"] for item in scored) / len(scored) if scored else 0.0
        detectable_failures = sum(item["detectable_failure_count"] for item in scored)
        categorized_failures = sum(item["categorized_failure_count"] for item in scored)
        detectable_failure_trajectories = sum(
            item["detectable_failure_count"] > 0 for item in scored
        )
        categorized_failure_trajectories = sum(
            item["detectable_failure_count"] > 0
            and item["categorized_failure_count"]
            == item["detectable_failure_count"]
            for item in scored
        )
        categorization_rate = (
            categorized_failure_trajectories / detectable_failure_trajectories
            if detectable_failure_trajectories
            else 1.0
        )

        callbacks.report_status(
            JobStatusUpdate(
                phase=JobPhase.POST_PROCESSING,
                status=JobStatus.RUNNING,
                message=MessageInfo(
                    message="Preparing evaluation result", message_code="post_processing"
                ),
            )
        )

        return JobResults(
            id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=config.model.name,
            results=[
                EvaluationResult(
                    metric_name="atif_overall_score",
                    metric_value=avg_score,
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="atif_trajectory_count",
                    metric_value=float(len(scored)),
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="atif_failure_categorization_rate",
                    metric_value=categorization_rate,
                    metric_type="float",
                ),
            ],
            overall_score=avg_score,
            num_examples_evaluated=len(scored),
            duration_seconds=time.monotonic() - start_time,
            completed_at=datetime.now(UTC),
            evaluation_metadata={
                "atif_trajectories": scored,
                "atif_scoring_mode": "auto",
                "atif_failure_threshold": failure_threshold,
                "atif_detectable_failure_count": detectable_failures,
                "atif_categorized_failure_count": categorized_failures,
                "atif_detectable_failure_trajectory_count": detectable_failure_trajectories,
                "atif_categorized_failure_trajectory_count": categorized_failure_trajectories,
                "atif_failure_categorization_rate": categorization_rate,
                "atif_training_threshold": training_threshold,
                "atif_training_manifest": eligible_paths,
            },
        )

    def _discover_atif_files(self, uri: str) -> list[Path]:
        path = Path(uri)
        if not path.exists():
            raise ATIFLoadError(f"ATIF path does not exist: {uri}")
        if path.is_file():
            if path.suffix.lower() != ".json":
                raise ATIFLoadError(f"ATIF input must be a JSON file: {path}")
            return [path]

        if not path.is_dir():
            raise ATIFLoadError(f"ATIF input is neither a file nor directory: {path}")
        return sorted(
            [p for p in path.rglob("*.json") if p.is_file()],
            key=lambda item: item.as_posix(),
        )

    @staticmethod
    def _supported_schema_versions() -> frozenset[str]:
        """Read supported versions from the SDK model without schema resolution."""
        # Trajectory is recursive, so Pydantic may represent its JSON schema as
        # a top-level ``$ref``. Inspecting the field annotation avoids relying
        # on the schema layout while keeping the SDK as the source of truth.
        annotation = Trajectory.model_fields["schema_version"].annotation
        return frozenset(str(version) for version in get_args(annotation))

    @classmethod
    def _validate_trajectory_limits(
        cls, trajectory: Trajectory, source: Path, max_steps_per_trajectory: int
    ) -> None:
        if max_steps_per_trajectory < 1:
            raise ATIFLoadError("max_steps_per_trajectory must be positive")

        def validate(current: Trajectory, location: str) -> None:
            if len(current.steps) > max_steps_per_trajectory:
                raise ATIFLoadError(
                    f"{source}: {location} contains {len(current.steps)} steps; "
                    f"maximum is {max_steps_per_trajectory}"
                )
            for index, subagent in enumerate(current.subagent_trajectories or []):
                validate(subagent, f"{location}.subagent_trajectories[{index}]")

        validate(trajectory, "trajectory")

    def _load_trajectories(
        self,
        files: list[Path],
        *,
        max_file_bytes: int = MAX_ATIF_FILE_BYTES,
        max_files: int = MAX_ATIF_FILES,
        max_steps_per_trajectory: int = MAX_STEPS_PER_TRAJECTORY,
    ) -> list[dict[str, Any]]:
        if max_file_bytes < 1:
            raise ATIFLoadError("max_file_bytes must be positive")
        if max_files < 1:
            raise ATIFLoadError("max_trajectory_files must be positive")
        if not files:
            raise ATIFLoadError("No ATIF JSON trajectory files were found")
        if len(files) > max_files:
            raise ATIFLoadError(
                f"ATIF input contains {len(files)} files; maximum is {max_files}"
            )

        trajectories: list[dict[str, Any]] = []
        trajectory_sources: dict[str, Path] = {}
        supported_versions = self._supported_schema_versions()
        for file in files:
            try:
                file_size = file.stat().st_size
                if file_size > max_file_bytes:
                    raise ATIFLoadError(
                        f"{file}: size {file_size} bytes exceeds maximum "
                        f"of {max_file_bytes} bytes"
                    )
                data = json.loads(file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ATIFLoadError("top-level JSON value must be an object")
                schema_version = data.get("schema_version", "ATIF-v1.8")
                if (
                    not isinstance(schema_version, str)
                    or schema_version not in supported_versions
                ):
                    supported = ", ".join(sorted(supported_versions))
                    raise ATIFLoadError(
                        f"unsupported schema_version {schema_version!r}; "
                        f"supported versions: {supported}"
                    )
                parsed = Trajectory.model_validate(data)
                self._validate_trajectory_limits(
                    parsed, file, max_steps_per_trajectory
                )
            except ATIFLoadError:
                raise
            except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
                raise ATIFLoadError(f"invalid ATIF trajectory {file}: {exc}") from exc

            if parsed.trajectory_id is not None:
                previous = trajectory_sources.get(parsed.trajectory_id)
                if previous is not None:
                    raise ATIFLoadError(
                        f"duplicate trajectory_id {parsed.trajectory_id!r} in "
                        f"{file}; already defined in {previous}"
                    )
                trajectory_sources[parsed.trajectory_id] = file
            trajectory = parsed.model_dump(mode="json")
            trajectory["_source_path"] = str(file)
            trajectories.append(trajectory)
        return trajectories

    async def _score_trajectories(
        self,
        trajectories: list[dict[str, Any]],
        concurrency_limit: int,
        *,
        failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    ) -> list[dict[str, Any]]:
        criteria = await self._derive_criteria(trajectories[0]) if trajectories else {"criteria": []}
        semaphore = asyncio.Semaphore(max(1, concurrency_limit))

        start = time.monotonic()
        first_done = False

        async def score_one(index: int, trajectory: dict[str, Any]) -> dict[str, Any]:
            nonlocal first_done
            async with semaphore:
                details = await self._score_single_trajectory_details(
                    trajectory, criteria, failure_threshold
                )
                if not first_done:
                    first_done = True
                    elapsed = time.monotonic() - start
                    remaining = elapsed * (len(trajectories) - 1)
                    logger.info("Estimated remaining time: %.2fs", remaining)
                return {
                    "trajectory_id": trajectory.get("trajectory_id", f"trajectory-{index}"),
                    "source_path": trajectory.get("_source_path"),
                    **details,
                    "step_count": len(trajectory.get("steps", [])),
                }

        tasks = [score_one(i, t) for i, t in enumerate(trajectories)]
        return list(await asyncio.gather(*tasks))

    async def _derive_criteria(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        extra = trajectory.get("extra") or {}
        prompt = {
            "task_instruction": extra.get("task_instruction", ""),
            "request": "Extract concise scoring criteria as JSON",
        }
        response = await self._judge_call(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"criteria": [{"name": "default", "weight": 1.0}]}

    async def _score_single_trajectory(self, trajectory: dict[str, Any], criteria: dict[str, Any]) -> float:
        details = await self._score_single_trajectory_details(
            trajectory, criteria, DEFAULT_FAILURE_THRESHOLD
        )
        return details["score"]

    async def _score_single_trajectory_details(
        self,
        trajectory: dict[str, Any],
        criteria: dict[str, Any],
        failure_threshold: float,
    ) -> dict[str, Any]:
        step_scores: list[float] = []
        step_results: list[dict[str, Any]] = []
        detectable_failure_count = 0
        categorized_failure_count = 0
        for step in trajectory.get("steps", []):
            payload = {
                "criteria": criteria,
                "step": step,
                "request": "Return JSON with numeric 'score' between 0 and 1",
            }
            response = await self._judge_call(payload)
            score = self._parse_score_response(
                response,
                trajectory_id=trajectory.get("trajectory_id"),
                step_id=step.get("step_id"),
            )
            step_scores.append(score)

            step_result: dict[str, Any] = {
                "step_id": step.get("step_id"),
                "score": score,
            }
            if score < failure_threshold:
                detectable_failure_count += 1
                category = await self._categorize_failure(
                    step, score, criteria
                )
                step_result.update(category)
                if category["category"] != "uncategorized":
                    categorized_failure_count += 1
            step_results.append(step_result)

        return {
            "score": sum(step_scores) / len(step_scores) if step_scores else 0.0,
            "steps": step_results,
            "detectable_failure_count": detectable_failure_count,
            "categorized_failure_count": categorized_failure_count,
        }

    @staticmethod
    def _parse_score_response(
        response: str,
        *,
        trajectory_id: Any,
        step_id: Any,
    ) -> float:
        """Parse and validate one judge score without masking protocol errors."""
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise JudgeResponseError(
                "judge returned non-JSON score response "
                f"for trajectory={trajectory_id!r}, step={step_id!r}"
            ) from exc

        if not isinstance(parsed, dict) or "score" not in parsed:
            raise JudgeResponseError(
                "judge score response is missing numeric 'score' "
                f"for trajectory={trajectory_id!r}, step={step_id!r}"
            )

        raw_score = parsed["score"]
        if isinstance(raw_score, bool):
            raise JudgeResponseError(
                "judge score must be numeric, not boolean "
                f"for trajectory={trajectory_id!r}, step={step_id!r}"
            )
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise JudgeResponseError(
                "judge score is not numeric "
                f"for trajectory={trajectory_id!r}, step={step_id!r}"
            ) from exc

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise JudgeResponseError(
                "judge score must be finite and between 0 and 1 "
                f"for trajectory={trajectory_id!r}, step={step_id!r}"
            )
        return score

    async def _categorize_failure(
        self, step: dict[str, Any], score: float, criteria: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {
            "criteria": criteria,
            "step": step,
            "score": score,
            "allowed_categories": sorted(FAILURE_CATEGORIES),
            "request": (
                "Classify this low-scoring agent step. Return only JSON with "
                "category, confidence, and rationale. category must be one of "
                "tool_selection_failure, context_loss, policy_boundary_violation, "
                "reasoning_failure, or none. Use none when no actionable "
                "failure is detectable."
            ),
        }
        raw_response = await self._judge_call(payload)
        try:
            parsed = json.loads(raw_response)
            category = parsed["category"]
            if category not in FAILURE_CATEGORIES:
                raise ValueError("unknown failure category")
            confidence = float(parsed.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            result = {
                "category": category,
                "confidence": confidence,
                "rationale": str(parsed.get("rationale", "")),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            result = {
                "category": "uncategorized",
                "confidence": 0.0,
                "rationale": "Judge response did not match the failure taxonomy",
                "raw_judge_response": raw_response,
            }
        return result

    async def _judge_call(self, payload: dict[str, Any]) -> str:
        retries = 3
        delay = 0.5
        credentials = resolve_model_credentials()
        headers = {}
        if credentials.api_key:
            api_key = (
                "api-key:ref"
                if os.getenv("EVALHUB_MODE") == "k8s"
                else credentials.api_key
            )
            headers["Authorization"] = f"Bearer {api_key}"
        request_body = {
            "model": self.job_spec.model.name,
            "messages": [{"role": "user", "content": json.dumps(payload)}],
        }
        for attempt in range(retries):
            start = time.monotonic()
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8080/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                )
            latency_ms = (time.monotonic() - start) * 1000
            logger.info("atif.judge.call.latency=%s", latency_ms)
            logger.info("atif.judge.call.count=1")
            logger.info("atif.judge.token.count=%s", len(response.text))
            if response.status_code == 429 and attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if response.is_error:
                logger.error(
                    "atif.judge.call.failed status=%s body=%s",
                    response.status_code,
                    response.text[:500],
                )
            response.raise_for_status()
            try:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                return response.text
            return content if isinstance(content, str) else json.dumps(content)
        raise RuntimeError("Judge call failed after retries")


def main() -> None:
    adapter = ATIFAdapter()
    callbacks = DefaultCallbacks.from_adapter(adapter)
    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
    callbacks.report_results(results)


if __name__ == "__main__":
    main()
