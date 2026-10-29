"""Sanity checks for the docs example scripts (they are executed in CI as notebooks)."""

import py_compile
import re
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"

# A hardcoded key would be a long quoted string assigned to a key argument.
KEY_LITERAL = re.compile(r'(?:api_key|edr_key)\s*=\s*"[^"\s]{20,}"')


def test_docs_examples_compile():
    examples = sorted(DOCS.glob("example_*.py"))
    assert examples, "no example scripts found in docs/"
    for example in examples:
        py_compile.compile(str(example), doraise=True)


def test_docs_examples_never_contain_a_literal_key():
    # The examples must read keys from the environment, never hardcode them.
    for example in sorted(DOCS.glob("example_*.py")):
        text = example.read_text(encoding="utf-8")
        assert not KEY_LITERAL.search(text), f"hardcoded key in {example.name}"
