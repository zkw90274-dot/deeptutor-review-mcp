"""Regenerate spec_schema.json from the Pydantic model.

Run after changing ReviewSpec:

    .venv/Scripts/python.exe scripts/update_spec_schema.py

The test suite compares the committed file with the live model, so a schema
edit without regenerating this artifact fails CI.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# The review package is normally installed as a wheel and `deeptutor` is a
# regular package, so a bare import always resolves to site-packages. Prepend
# the checkout to the package's own __path__ (the same trick tests/conftest.py
# uses) so this artifact is generated from the working tree, not the last wheel.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import deeptutor  # noqa: E402

_CHECKOUT_PACKAGE = SRC / "deeptutor"
if str(_CHECKOUT_PACKAGE) not in deeptutor.__path__:
    deeptutor.__path__.insert(0, str(_CHECKOUT_PACKAGE))

from deeptutor.review.models import ReviewSpec  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "spec_schema.json"


def main() -> None:
    schema = ReviewSpec.model_json_schema()
    OUT.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
