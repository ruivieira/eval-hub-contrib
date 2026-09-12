# ATIF EvalHub Adapter

The ATIF adapter evaluates pre-recorded [Agent Trajectory Interchange Format
(ATIF)](https://github.com/agent-trajectory/atif) trajectories with an
OpenAI-compatible language-model judge. It is an EvalHub provider that loads
ATIF JSON files, derives evaluation criteria, scores trajectory steps, and
publishes aggregate scores and failure-categorization metadata.

This adapter is currently an auto-scoring proof of concept. It supports local
files and mounted directories. S3-backed input discovery, benchmark/reference
scoring, custom rubrics, nested subagent scoring, typed training manifests, and
report attachments are not implemented by this adapter.

## How it works

For each EvalHub job, the adapter:

1. Reads the job parameters and validates the configured thresholds.
2. Discovers one JSON file or recursively discovers JSON files in a directory.
3. Parses each file with the ATIF models from `eval-hub-sdk`.
4. Validates the ATIF schema version, duplicate trajectory IDs, file size,
   collection size, and step limits.
5. Calls the runtime-sidecar judge to derive criteria from the first trajectory.
6. Scores every top-level step in every trajectory concurrently.
7. Requests a failure category for steps whose score is below
   `failure_threshold`.
8. Aggregates step scores into a trajectory score and then an overall score.
9. Optionally marks trajectories as training-eligible when their score meets
   `training_threshold`.
10. Returns EvalHub metrics and detailed generic evaluation metadata.

The adapter calls the judge through the runtime sidecar at:

```text
http://localhost:8080/v1/chat/completions
```

The judge receives JSON in the OpenAI chat message content. The adapter does
not make the configured `JobSpec.model.url` request directly; the sidecar is
responsible for routing and credentials.

## Input format

`trajectory_path` may point to:

- a single `.json` ATIF trajectory file; or
- a directory, in which case all `*.json` files are discovered recursively in
  deterministic path order.

The adapter uses the SDK's `Trajectory` model for validation. It currently
accepts the schema versions exposed by the installed SDK, including
`ATIF-v1.0` through `ATIF-v1.8` in the validated environment.

Each trajectory must contain valid ATIF data. Duplicate non-null
`trajectory_id` values across files are rejected. Empty collections, malformed
JSON, unsupported schema versions, oversized files, and trajectories exceeding
the step limit fail the job during loading.

The default path is `/test_data/trajectory.json`, which is suitable for a
mounted EvalHub test-data volume. For local execution, provide a path relative
to the adapter's working directory or an absolute path.

## Parameters

Parameters are supplied through the EvalHub job's generic `parameters` object.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `trajectory_path` | string | `/test_data/trajectory.json` | Single ATIF JSON file or directory containing ATIF JSON files. |
| `concurrency_limit` | integer | `10` | Maximum number of top-level trajectories scored concurrently. Values below 1 are clamped to one for scoring. |
| `max_file_bytes` | integer | `10485760` | Maximum size of an individual input file in bytes. Must be positive. |
| `max_trajectory_files` | integer | `10000` | Maximum number of discovered JSON files. Must be positive. |
| `max_steps_per_trajectory` | integer | `500` | Maximum number of steps in each trajectory. The validation also checks nested subagent trajectories, although nested trajectories are not currently scored recursively. |
| `failure_threshold` | float | `0.5` | Scores strictly below this value are treated as detectable failures and sent for categorization. Must be in `[0, 1]`. |
| `training_threshold` | float | unset | Optional score threshold for training eligibility. A trajectory is eligible when its aggregate score is greater than or equal to this value. Must be in `[0, 1]`. |

The adapter rejects an explicitly supplied invalid threshold before loading
input. If `training_threshold` is omitted, `training_eligible` is `null` for
each trajectory and the training manifest is empty.

### Example parameters

```json
{
  "trajectory_path": "/test_data/atif",
  "concurrency_limit": 4,
  "max_file_bytes": 10485760,
  "max_trajectory_files": 1000,
  "max_steps_per_trajectory": 500,
  "failure_threshold": 0.5,
  "training_threshold": 0.8
}
```

## Scoring behavior

### Criteria derivation

The adapter sends the first trajectory's `extra.task_instruction` to the
judge and asks for concise JSON criteria. If that response is not valid JSON,
the adapter uses a fallback criterion named `default` with weight `1.0`.

### Step scoring

Each top-level step is sent to the judge with the derived criteria. The judge
must return JSON containing a numeric `score` in the inclusive range `[0, 1]`.
Scores must be finite; booleans, missing values, non-numeric values, malformed
JSON, and out-of-range values are rejected.

A genuine score of `0.0` is valid. Invalid score responses raise
`JudgeResponseError` and fail the evaluation rather than being silently
converted to zero.

The trajectory score is the arithmetic mean of its top-level step scores. An
empty `steps` list currently produces a score of `0.0`; this behavior remains
part of the proof-of-concept contract and should be distinguished from a judge
protocol failure.

### Failure categorization

When a step score is below `failure_threshold`, the adapter makes a second
judge request for a category. Supported categories are:

- `tool_selection_failure`
- `context_loss`
- `policy_boundary_violation`
- `reasoning_failure`
- `none`

Invalid categorization responses are retained as `uncategorized` with the raw
response in the step metadata. Categorization rate
is calculated over trajectories containing detectable failures: a trajectory
counts as fully categorized only when all of its detectable failures receive a
category other than `uncategorized`.

The adapter currently scores top-level steps only. It validates nested
subagent step limits but does not recursively score or aggregate nested
subagent trajectories.

### Retries and errors

HTTP 429 responses are retried up to three attempts with exponential backoff.
Other HTTP errors, request failures, invalid score responses, and input
validation errors fail the job. A malformed judge response is therefore
visible as an evaluation failure instead of producing a misleading successful
zero score.

## Results

The adapter returns the following `EvaluationResult` metrics:

| Metric | Meaning |
| --- | --- |
| `atif_overall_score` | Mean score across all loaded trajectories. |
| `atif_trajectory_count` | Number of scored trajectories. |
| `atif_failure_categorization_rate` | Fraction of trajectories with detectable failures for which every detectable failure was categorized; `1.0` when no detectable failures exist. |

The EvalHub `overall_score` is set to `atif_overall_score`, and
`num_examples_evaluated` is the number of loaded trajectories.

## Evaluation metadata

`evaluation_metadata` includes:

```text
atif_trajectories
atif_scoring_mode
atif_failure_threshold
atif_detectable_failure_count
atif_categorized_failure_count
atif_detectable_failure_trajectory_count
atif_categorized_failure_trajectory_count
atif_failure_categorization_rate
atif_training_threshold
atif_training_manifest
```

Each entry in `atif_trajectories` contains the trajectory ID, local source
path, aggregate score, step count, per-step results, detectable-failure count,
categorized-failure count, and `training_eligible`. Each step result contains
its step ID and score; categorized failures also contain category, confidence,
and rationale.

When `training_threshold` is set, `atif_training_manifest` contains the local
source paths of trajectories whose aggregate score meets the threshold. These
are generic metadata fields. A typed SDK `training_manifest` field, original
S3 paths, and a downloadable report attachment require downstream EvalHub and
SDK support.

## Provider and container

The provider ID is `atif`, and the default benchmark ID is
`atif_trajectory_default`. The provider manifest is in [`provider.yaml`](provider.yaml).
The adapter image is built from [`Containerfile`](Containerfile) and runs
`python main.py` as a non-root user.

The provider's benchmark pass criterion is an `atif_overall_score` of at least
`0.5`. This benchmark criterion is separate from `failure_threshold` and
`training_threshold`:

- `failure_threshold` controls failure categorization.
- `training_threshold` controls training eligibility metadata.
- The benchmark pass criterion controls the benchmark's pass/fail evaluation.

## Local development and tests

From this adapter directory, install the runtime and test requirements, then
run the tests:

```sh
pip install -r requirements.txt -r requirements-test.txt
PYTHONPATH=. pytest -q tests
```

The test suite covers ATIF parsing and limits, supported schema versions,
duplicate IDs, malformed input, score validation, genuine zero scores,
failure categorization, 429 retries, training eligibility, and invalid
training thresholds.

## Current limitations

- Input is local or mounted filesystem data; S3 discovery and download are
  owned by a downstream EvalHub input contract.
- Only the `auto` scoring flow is implemented.
- Benchmark/reference scoring and custom rubric configuration are not
  implemented.
- Nested `subagent_trajectories` are validated for limits but not scored.
- Partial-result and resume/checkpoint policies are not implemented.
- Per-step judge diagnostics are limited; full structured redacted diagnostics
  and typed result fields require additional work.
- Training eligibility is exposed through generic metadata only. It does not
  yet create a typed SDK field or report attachment.
- Production image promotion, dependency onboarding, and release management
  are outside this adapter directory.
