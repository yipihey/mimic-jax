# SHARK prescription data

`cloudy_cie.json` is a mechanically packaged copy of the Cloudy CIE cooling
grid distributed by [ICRAR/SHARK](https://github.com/ICRAR/shark). It was
generated with `scripts/import_shark_cooling_table.py` from pinned revision
`5af50d8fa7a040883409b10171c645e1db4e5fb2`. The JSON provenance block records
the SHA-256 checksum of every source table.

The scientific values are unchanged: only comments and unused electron,
hydrogen, and total-particle columns are omitted. The package is distributed
under GPL-3.0-or-later, matching `pyproject.toml` and upstream SHARK. Regenerate
the artifact rather than editing it by hand when the pinned upstream revision
changes.
