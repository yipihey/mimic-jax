#!/usr/bin/env python3
"""Compile the SHARK rate oracle against a pinned upstream checkout."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shark-source", required=True, type=Path)
    parser.add_argument("--shark-build", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-directory", action="append", default=[], type=Path)
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    source = repository / "tests/mimic_jax/upstream/shark_rate_oracle.cpp"
    library = arguments.shark_build / "libshark.dylib"
    if not library.is_file():
        raise FileNotFoundError(f"Build upstream SHARK first; missing {library}")

    with tempfile.TemporaryDirectory(prefix="mimic-shark-oracle-") as temporary:
        executable = Path(temporary) / "shark_rate_oracle"
        command = [
            "c++",
            "-std=c++17",
            "-O2",
            f"-I{arguments.shark_source / 'include'}",
            *(
                f"-I{path}"
                for path in (
                    tuple(arguments.include_directory)
                    + (
                        (Path("/opt/homebrew/include"),)
                        if Path("/opt/homebrew/include").is_dir()
                        else ()
                    )
                )
            ),
            str(source),
            f"-L{arguments.shark_build}",
            "-lshark",
            *(
                ("-L/opt/homebrew/lib", "-lgsl", "-lgslcblas")
                if Path("/opt/homebrew/lib").is_dir()
                else ()
            ),
            f"-Wl,-rpath,{arguments.shark_build}",
            "-o",
            str(executable),
        ]
        subprocess.run(command, check=True)
        completed = subprocess.run(
            [str(executable), str(arguments.config)],
            cwd=arguments.shark_source,
            check=True,
            capture_output=True,
            text=True,
        )
    begin = "MIMIC_SHARK_ORACLE_BEGIN\n"
    end = "\nMIMIC_SHARK_ORACLE_END"
    try:
        oracle_json = completed.stdout.split(begin, maxsplit=1)[1].split(end, maxsplit=1)[0]
    except IndexError as error:
        raise RuntimeError("Upstream SHARK oracle did not emit its JSON sentinels") from error
    payload = json.loads(oracle_json)
    config_bytes = arguments.config.read_bytes()
    payload["provenance"] = {
        "upstream_repository": "https://github.com/ICRAR/shark",
        "upstream_revision": "5af50d8fa7a040883409b10171c645e1db4e5fb2",
        "harness": source.relative_to(repository).as_posix(),
        "config_name": arguments.config.name,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
