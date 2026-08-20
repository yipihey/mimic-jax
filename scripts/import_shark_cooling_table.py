#!/usr/bin/env python3
"""Import an upstream SHARK cooling-table family with provenance.

This is a data-packaging step, not part of a simulation.  The generated JSON
is small, human-inspectable, and consumed as immutable JAX constants by the
cooling prescription.
"""

import argparse
import hashlib
import json
from pathlib import Path


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shark-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", default="C08.00", choices=("C08.00", "S93"))
    parser.add_argument(
        "--revision",
        default="5af50d8fa7a040883409b10171c645e1db4e5fb2",
    )
    return parser.parse_args()


def _data_lines(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped


def main():
    arguments = _arguments()
    directory = arguments.shark_source / "data/cooling"
    index_path = directory / f"{arguments.family}_tables.txt"
    entries = []
    source_hashes = {}
    for line in _data_lines(index_path):
        metallicity_text, filename = line.split()
        entries.append((float(metallicity_text), filename))
    source_hashes[index_path.name] = hashlib.sha256(index_path.read_bytes()).hexdigest()

    temperatures = None
    cooling_functions = []
    for _metallicity, filename in entries:
        path = directory / filename
        rows = [tuple(float(value) for value in line.split()) for line in _data_lines(path)]
        row_temperatures = [row[0] for row in rows]
        if temperatures is None:
            temperatures = row_temperatures
        elif row_temperatures != temperatures:
            raise ValueError(f"Temperature grid differs in {path}")
        cooling_functions.append([row[4] for row in rows])
        source_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

    payload = {
        "provenance": {
            "upstream_repository": "https://github.com/ICRAR/shark",
            "upstream_revision": arguments.revision,
            "family": arguments.family,
            "source_sha256": source_hashes,
        },
        "log10_temperature_k": temperatures,
        "metallicity": [entry[0] for entry in entries],
        "log10_cooling_function": cooling_functions,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
