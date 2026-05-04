"""Validate that every new adapter directory in a PR contains a valid provider.yaml."""

import os
import re
import subprocess
import sys

import yaml

SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def get_added_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", f"origin/{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def is_preexisting_dir(name: str, base_ref: str) -> bool:
    ls = subprocess.run(
        ["git", "ls-tree", f"origin/{base_ref}", "--", f"adapters/{name}/"],
        capture_output=True,
        text=True,
    )
    return bool(ls.stdout.strip())


def new_adapter_dirs(added_files: list[str], base_ref: str) -> list[str]:
    candidates: set[str] = set()
    for path in added_files:
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] == "adapters":
            name = parts[1]
            if not SAFE_NAME.match(name):
                print(f"::warning::Skipping suspicious adapter dir name: {name!r}")
                continue
            candidates.add(name)

    new = []
    for name in sorted(candidates):
        if is_preexisting_dir(name, base_ref):
            print(f"adapters/{name}: pre-existing directory, skipping.")
        else:
            new.append(name)
    return new


def validate(name: str) -> bool:
    print(f"adapters/{name}: new adapter detected, checking provider.yaml …")
    provider_yaml = f"adapters/{name}/provider.yaml"

    if not os.path.isfile(provider_yaml):
        print(
            f"::error file={provider_yaml}::Missing provider.yaml in adapters/{name}/. "
            "Every new adapter must include a provider.yaml. "
            "See CONTRIBUTING.md for the required schema."
        )
        return False

    try:
        with open(provider_yaml) as f:
            yaml.safe_load(f)
        print(f"adapters/{name}: provider.yaml is valid YAML ✓")
        return True
    except yaml.YAMLError as exc:
        print(f"::error file={provider_yaml}::Invalid YAML in adapters/{name}/provider.yaml: {exc}")
        return False


def main() -> int:
    base_ref = os.environ.get("BASE_REF")
    if not base_ref:
        print("::error::BASE_REF environment variable is not set.")
        return 1

    added_files = get_added_files(base_ref)
    new_dirs = new_adapter_dirs(added_files, base_ref)

    if not new_dirs:
        print("No new adapter directories detected — nothing to validate.")
        return 0

    errors = [name for name in new_dirs if not validate(name)]

    if errors:
        return 1

    print("All new adapters passed provider.yaml validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
