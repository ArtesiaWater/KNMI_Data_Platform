"""Sanity checks for the docs example notebooks (they are executed in CI)."""

import json
import re
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"

# A hardcoded key would be a long quoted string assigned to a key argument.
KEY_LITERAL = re.compile(r"(?:api_key|edr_key)\s*=\s*\"[^\"\s]{20,}\"")


def test_docs_examples_are_valid_notebooks():
    examples = sorted(DOCS.glob("example_*.ipynb"))
    assert examples, "no example notebooks found in docs/"
    for example in examples:
        nb = json.loads(example.read_text(encoding="utf-8"))
        assert nb.get("cells"), f"no cells in {example.name}"


def test_docs_examples_never_contain_a_literal_key():
    # The examples must read keys from the environment, never hardcode them.
    for example in sorted(DOCS.glob("example_*.ipynb")):
        text = example.read_text(encoding="utf-8")
        assert not KEY_LITERAL.search(text), f"hardcoded key in {example.name}"
