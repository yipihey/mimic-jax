"""Configure JAX precision before importing the SAGE16 package under test."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
