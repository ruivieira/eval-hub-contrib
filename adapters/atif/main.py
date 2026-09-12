#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


class ATIFLoadError(ValueError):
    """Raised when an ATIF input collection cannot be loaded safely."""


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
        scored = asyncio.run(self._score_trajectories(trajectories, concurrency_limit))

        avg_score = sum(item["score"] for item in scored) / len(scored) if scored else 0.0

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
            ],
            overall_score=avg_score,
            num_examples_evaluated=len(scored),
            duration_seconds=time.monotonic() - start_time,
            completed_at=datetime.now(UTC),
            evaluation_metadata={
                "atif_trajectories": scored,
                "atif_scoring_mode": "auto",
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
        """Read supported versions from the SDK model instead of duplicating them."""
        schema = Trajectory.model_json_schema()
        versions = schema["properties"]["schema_version"].get("enum", [])
        return frozenset(str(version) for version in versions)

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
            trajectories.append(parsed.model_dump(mode="json"))
        return trajectories

    async def _score_trajectories(self, trajectories: list[dict[str, Any]], concurrency_limit: int) -> list[dict[str, Any]]:
        criteria = await self._derive_criteria(trajectories[0]) if trajectories else {"criteria": []}
        semaphore = asyncio.Semaphore(max(1, concurrency_limit))

        start = time.monotonic()
        first_done = False

        async def score_one(index: int, trajectory: dict[str, Any]) -> dict[str, Any]:
            nonlocal first_done
            async with semaphore:
                score = await self._score_single_trajectory(trajectory, criteria)
                if not first_done:
                    first_done = True
                    elapsed = time.monotonic() - start
                    remaining = elapsed * (len(trajectories) - 1)
                    logger.info("Estimated remaining time: %.2fs", remaining)
                return {
                    "trajectory_id": trajectory.get("trajectory_id", f"trajectory-{index}"),
                    "score": score,
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
        step_scores: list[float] = []
        for step in trajectory.get("steps", []):
            payload = {
                "criteria": criteria,
                "step": step,
                "request": "Return JSON with numeric 'score' between 0 and 1",
            }
            response = await self._judge_call(payload)
            try:
                parsed = json.loads(response)
                step_scores.append(float(parsed.get("score", 0.0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                step_scores.append(0.0)
        return sum(step_scores) / len(step_scores) if step_scores else 0.0

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
