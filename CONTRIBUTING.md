# Contributing

This repository contains community-contributed adapters that integrate evaluation frameworks with [eval-hub](https://github.com/eval-hub/eval-hub). Each adapter implements the `FrameworkAdapter` pattern from the evalhub-sdk.

## Adding an adapter

1. Create a new directory under `adapters/` named after the framework (e.g. `adapters/my-framework/`)
2. Add a `provider.yaml` describing the provider — CI will reject new adapters without one (see [Provider definition](#provider-definition-provideryaml) below)
3. Implement `main.py` using the `FrameworkAdapter` pattern from evalhub-sdk
4. Add a `Containerfile` and `requirements.txt`
5. Add build/push targets to the root `Makefile`
6. Document the adapter in its own `README.md`

## Provider definition (`provider.yaml`)

Every adapter directory **must** contain a `provider.yaml`. CI checks for its presence and validates that it is well-formed YAML on every pull request that adds a new adapter.

The file describes the provider to eval-hub: its identity, runtime resource requirements, and the benchmarks it exposes. The top-level fields are:

| Field | Required | Description |
|---|---|---|
| `id` | ✓ | Unique identifier used in API calls (e.g. `mteb`) |
| `name` | ✓ | Human-readable display name |
| `description` | ✓ | Short description of what the adapter evaluates |
| `type` | ✓ | `builtin` for community adapters |
| `runtime.k8s.image` | ✓ | Container image for Kubernetes jobs |
| `runtime.k8s.entrypoint` | ✓ | Command to run inside the container |
| `runtime.k8s.cpu_request` / `memory_request` | ✓ | Kubernetes resource requests |
| `runtime.k8s.cpu_limit` / `memory_limit` | ✓ | Kubernetes resource limits |
| `benchmarks` | ✓ | List of benchmark definitions (id, name, description, category, metrics, tags) |
| `parameters` | — | Optional list of configurable parameters with types and defaults |

See [`adapters/mteb/provider.yaml`](adapters/mteb/provider.yaml) and [`adapters/clear/provider.yaml`](adapters/clear/provider.yaml) for complete, annotated examples.

## Building adapters

```sh
# Build a specific adapter
make image-lighteval
make image-guidellm

# Build all adapters
make images

# Push to a registry
make push-lighteval REGISTRY=quay.io/your-org VERSION=v1.0.0
make push-guidellm REGISTRY=quay.io/your-org VERSION=v1.0.0
```

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org). All commit messages must follow the format:

```
<type>(<scope>): <subject>
```

**Common types:** `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`

**Examples:**

```
feat(lighteval): add support for custom task configuration
fix(guidellm): handle connection timeout on benchmark run
docs: update adapter development guide
chore: bump dependencies
```

PRs targeting `main` will fail CI if any commit message does not follow this format.

If you have [pre-commit](https://pre-commit.com) installed, commit messages are also checked locally:

```sh
pre-commit install --hook-type commit-msg
```
