"""Managed execution of the pinned native SHARK population reference.

The native executable is the authoritative topology/event oracle while the
independent JAX implementation is validated.  Scientific computation never
downloads or builds SHARK implicitly: paths, revisions, checksums, seed, and
the effective configuration are explicit durable provenance.
"""

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from mimic_jax.shark.types import SHARK_UPSTREAM_REVISION

PUBLIC_CI_TREE_SHA256 = "c072a937941fefb9aac441fc319ff030ceb666af4a07f1b88c0f02c5d76a3f43"
PUBLIC_CI_REDSHIFTS_SHA256 = "816a885a6e73d6d9022fffeb8667acfe2b0719a6cb0da2d696abe61500b135b9"


@dataclass(frozen=True)
class SharkReferenceRun:
    """Paths and provenance returned by a completed upstream execution."""

    executable: str
    effective_config: str
    output_directory: str
    catalogue: str
    upstream_revision: str
    seed: int
    tree_sha256: str
    redshift_sha256: str
    config_sha256: str
    started_at_utc: str
    elapsed_seconds: float
    command: tuple[str, ...]
    simulation_name: str

    def write_manifest(self, path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return target


def sha256_file(path, *, chunk_size=1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a local input or artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path, expected) -> str:
    """Verify an input checksum and return the observed digest."""

    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}")
    return observed


def upstream_git_revision(source_directory) -> tuple[str, bool]:
    """Return the checked-out revision and dirty status of an upstream clone."""

    source = Path(source_directory)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            # An out-of-tree or unignored build directory does not change the
            # reference source. Tracked modifications do, and remain fatal.
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def require_pinned_upstream(source_directory, *, allow_dirty=False) -> None:
    """Reject an unreviewed native reference source tree."""

    revision, dirty = upstream_git_revision(source_directory)
    if revision != SHARK_UPSTREAM_REVISION:
        raise ValueError(f"Expected SHARK revision {SHARK_UPSTREAM_REVISION}, found {revision}")
    if dirty and not allow_dirty:
        raise ValueError(
            "Pinned SHARK source tree is dirty; pass allow_dirty=True only deliberately"
        )


def _replace_ini_values(text: str, replacements: Mapping[tuple[str, str], str]) -> str:
    section = None
    consumed = set()
    output = []
    for line in text.splitlines():
        match = re.match(r"\s*\[([^]]+)\]\s*$", line)
        if match:
            section = match.group(1).strip()
            output.append(line)
            continue
        option = re.match(r"(\s*)([A-Za-z0-9_]+)(\s*=).*$", line)
        key = (section, option.group(2)) if option and section else None
        if key in replacements:
            output.append(
                f"{option.group(1)}{option.group(2)}{option.group(3)} {replacements[key]}"
            )
            consumed.add(key)
        else:
            output.append(line)
    missing = set(replacements) - consumed
    if missing:
        raise ValueError(f"Template is missing required SHARK options: {sorted(missing)}")
    return "\n".join(output) + "\n"


def prepare_reference_config(
    template,
    destination,
    *,
    tree_file,
    redshift_file,
    output_directory,
    seed=123456,
    model_name="lagos23-reference",
    simulation_batch=0,
) -> Path:
    """Materialize the effective native config without mutating the template."""

    tree = Path(tree_file).resolve()
    match = re.match(r"(.+)\.[0-9]+\.hdf5$", str(tree))
    if not match:
        raise ValueError("tree_file must end in '.<subvolume>.hdf5'")
    replacements = {
        ("execution", "seed"): str(int(seed)),
        ("execution", "simulation_batches"): str(int(simulation_batch)),
        ("execution", "output_directory"): str(Path(output_directory).resolve()),
        ("execution", "name_model"): model_name,
        ("simulation", "tree_files_prefix"): match.group(1),
        ("simulation", "redshift_file"): str(Path(redshift_file).resolve()),
    }
    rendered = _replace_ini_values(Path(template).read_text(encoding="utf-8"), replacements)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination


def run_reference_shark(
    *,
    executable,
    config,
    output_directory,
    tree_file,
    redshift_file,
    snapshot=199,
    subvolume=0,
    model_name="lagos23-reference",
    simulation_name="mini-SURFS",
    seed=123456,
    source_directory: Optional[Path] = None,
    expected_tree_sha256=PUBLIC_CI_TREE_SHA256,
    expected_redshift_sha256=PUBLIC_CI_REDSHIFTS_SHA256,
    timeout_seconds=3600,
) -> SharkReferenceRun:
    """Run and validate one complete native SHARK reference population."""

    executable = Path(executable).resolve()
    config = Path(config).resolve()
    output = Path(output_directory).resolve()
    if source_directory is not None:
        require_pinned_upstream(source_directory)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    tree_digest = verify_sha256(tree_file, expected_tree_sha256)
    redshift_digest = verify_sha256(redshift_file, expected_redshift_sha256)
    output.mkdir(parents=True, exist_ok=True)
    command = (str(executable), str(config))
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    (output / "shark.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "shark.stderr.log").write_text(completed.stderr, encoding="utf-8")
    catalogue = (
        output / simulation_name / model_name / str(snapshot) / str(subvolume) / "galaxies.hdf5"
    )
    if not catalogue.is_file():
        raise RuntimeError(f"SHARK completed without expected catalogue {catalogue}")
    run = SharkReferenceRun(
        executable=str(executable),
        effective_config=str(config),
        output_directory=str(output),
        catalogue=str(catalogue),
        upstream_revision=SHARK_UPSTREAM_REVISION,
        seed=int(seed),
        tree_sha256=tree_digest,
        redshift_sha256=redshift_digest,
        config_sha256=sha256_file(config),
        started_at_utc=started.isoformat(),
        elapsed_seconds=elapsed,
        command=command,
        simulation_name=simulation_name,
    )
    run.write_manifest(output / "shark-reference-manifest.json")
    return run
