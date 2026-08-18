"""Automatic, explicit provenance capture for reproducible run reports."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from mimic_jax.reporting.model import Provenance, ProvenanceFile


def _run_git(repository: Path, *arguments: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, repository: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def checksummed_file(path, *, role: str, repository) -> ProvenanceFile:
    """Describe one input using a stable path, byte size, and SHA-256 digest."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Provenance input does not exist: {source}")
    repository = Path(repository)
    return ProvenanceFile(
        path=_display_path(source, repository),
        role=role,
        sha256=_file_sha256(source),
        size_bytes=source.stat().st_size,
    )


def _package_versions(names: Sequence[str]) -> Mapping[str, str]:
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _hardware(include_jax_runtime: bool) -> Mapping[str, Any]:
    values = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported",
    }
    if include_jax_runtime:
        try:
            import jax

            values["jax_backend"] = jax.default_backend()
            values["jax_devices"] = [str(device) for device in jax.devices()]
        except (ImportError, RuntimeError) as error:
            values["jax_runtime"] = f"unavailable: {error}"
    return values


def load_upstream_run_record(path) -> Mapping[str, Any]:
    """Load MIMIC's run-local ``version_info.json`` without replacing it as source of truth."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read upstream MIMIC run record {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Upstream MIMIC run record must be a JSON object: {source}")
    return payload


def capture_provenance(
    *,
    repository,
    command: Optional[Sequence[str]] = None,
    configuration_paths: Sequence[Any] = (),
    input_paths: Sequence[Any] = (),
    random_seeds: Optional[Mapping[str, Any]] = None,
    upstream_version_info=None,
    generated_at: Optional[str] = None,
    include_jax_runtime: bool = True,
    package_names: Sequence[str] = (
        "mimic-jax",
        "jax",
        "jaxlib",
        "numpy",
        "h5py",
        "matplotlib",
        "PyYAML",
    ),
) -> Provenance:
    """Capture git, files, software, backend, and command metadata for one run.

    Paths within the repository are stored relative to it. Callers choose which
    scientific inputs merit checksums, avoiding an expensive implicit walk over
    merger-tree datasets.
    """

    repository = Path(repository).resolve()
    if generated_at is None:
        generated_at = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
    resolved_command: Tuple[str, ...] = tuple(sys.argv if command is None else command)
    files = tuple(
        checksummed_file(path, role="configuration", repository=repository)
        for path in configuration_paths
    ) + tuple(checksummed_file(path, role="input", repository=repository) for path in input_paths)

    git_commit = _run_git(repository, "rev-parse", "HEAD")
    git_branch = _run_git(repository, "branch", "--show-current")
    git_status = _run_git(repository, "status", "--porcelain")
    git_dirty = None if git_status is None else bool(git_status)
    upstream_run = (
        {} if upstream_version_info is None else load_upstream_run_record(upstream_version_info)
    )
    return Provenance(
        generated_at=generated_at,
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        command=resolved_command,
        files=files,
        software=_package_versions(package_names),
        hardware=_hardware(include_jax_runtime),
        random_seeds=dict(random_seeds or {}),
        upstream_run=upstream_run,
    )
