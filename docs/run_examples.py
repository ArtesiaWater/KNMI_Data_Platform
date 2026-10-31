"""Execute the example notebooks in this directory in place.

Used by `tox -e notebooks` and the documentation workflow: the executed
notebooks (with outputs) are what Sphinx/nbsphinx renders into the docs.
API keys come from the environment (KNMI_OPENDATA_API_KEY / KNMI_EDR_API_KEY)
or from a gitignored .env file next to this script.
"""

import sys
from pathlib import Path

import papermill as pm

DOCS = Path(__file__).resolve().parent


def main() -> int:
    """Execute every example_*.ipynb notebook in place, return exit status."""
    notebooks = sorted(DOCS.glob("example_*.ipynb"))
    if not notebooks:
        print("no example notebooks found", file=sys.stderr)
        return 1
    status = 0
    for nb in notebooks:
        print(f"::group::Executing {nb.name}")
        try:
            pm.execute_notebook(str(nb), str(nb), kernel_name="python3")
        except pm.PapermillException as err:
            print(f"FAILED {nb.name}: {type(err).__name__}: {err}", file=sys.stderr)
            status = 1
        print("::endgroup::")
    return status


if __name__ == "__main__":
    sys.exit(main())
