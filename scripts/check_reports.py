#!/usr/bin/env python3
"""Validate committed run-report manifests, assets, checksums, and Markdown pairs."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

REPORT_SCHEMA_VERSION = "mimic-jax-report/v1"
VALID_STATUSES = {"passed", "warning", "failed", "not_evaluated"}
REPOSITORY = Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=REPOSITORY / "reports",
        help="report tree to validate (default: repository reports directory)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_artifacts(diagnostic):
    yield from diagnostic.get("artifacts", ())


def report_artifacts(payload):
    yield from payload.get("headline_artifacts", ())
    for diagnostic in payload.get("health", ()):
        yield from diagnostic_artifacts(diagnostic)
    for section in payload.get("sections", ()):
        yield from section.get("artifacts", ())
        for diagnostic in section.get("diagnostics", ()):
            yield from diagnostic_artifacts(diagnostic)


def report_diagnostics(payload):
    yield from payload.get("health", ())
    for section in payload.get("sections", ()):
        yield from section.get("diagnostics", ())


def validate_manifest(path: Path):
    errors = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"{path}: invalid JSON: {error}"]
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        errors.append(f"{path}: unsupported or missing schema_version")
    valid_kinds = ("run", "comparison", "multi_model_comparison")
    if payload.get("kind") not in valid_kinds:
        errors.append(f"{path}: kind must be 'run', 'comparison', or 'multi_model_comparison'")
    if payload.get("kind") == "multi_model_comparison":
        run_keys = tuple(run.get("key") for run in payload.get("runs", ()))
        if len(run_keys) < 3 or len(run_keys) != len(set(run_keys)) or None in run_keys:
            errors.append(f"{path}: multi-model report requires at least three unique run keys")
        for metric in payload.get("metrics", ()):
            metric_keys = tuple(value.get("model_key") for value in metric.get("values", ()))
            if len(metric_keys) != len(set(metric_keys)) or set(metric_keys) != set(run_keys):
                errors.append(
                    f"{path}: multi-model metric {metric.get('key', '<unknown>')} does not "
                    "provide exactly one value for every run"
                )
    markdown = path.parent / "index.md"
    if not markdown.is_file():
        errors.append(f"{path}: matching index.md is missing")

    for diagnostic in report_diagnostics(payload):
        status = diagnostic.get("status")
        if status not in VALID_STATUSES:
            errors.append(
                f"{path}: diagnostic {diagnostic.get('key', '<unknown>')} has invalid status "
                f"{status!r}"
            )
    checked_paths = set()
    for artifact in report_artifacts(payload):
        relative = artifact.get("path", "")
        candidate = PurePosixPath(relative)
        if not relative or candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"{path}: unsafe artifact path {relative!r}")
            continue
        if relative in checked_paths:
            continue
        checked_paths.add(relative)
        artifact_path = path.parent / relative
        if not artifact_path.is_file():
            errors.append(f"{path}: missing artifact {relative}")
            continue
        expected_size = artifact.get("size_bytes")
        if expected_size != artifact_path.stat().st_size:
            errors.append(f"{path}: size mismatch for {relative}")
        expected_checksum = artifact.get("sha256")
        if expected_checksum != sha256(artifact_path):
            errors.append(f"{path}: SHA-256 mismatch for {relative}")
    return errors


def main():
    arguments = parse_arguments()
    manifests = sorted(arguments.root.glob("**/report.json"))
    if not manifests:
        raise SystemExit(f"No report.json manifests found under {arguments.root}")
    errors = []
    for manifest in manifests:
        errors.extend(validate_manifest(manifest))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"Report validation failed with {len(errors)} error(s)")
    print(f"Validated {len(manifests)} report manifest(s) under {arguments.root}")


if __name__ == "__main__":
    main()
