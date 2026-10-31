"""Execute the example notebooks in this directory in place.

Used by `tox -e notebooks` and the documentation workflow: the executed
notebooks (with outputs) are what Sphinx/nbsphinx renders into the docs.
Cell errors (e.g. KNMIHTTPError when the KNMI API is down) do not fail the
run: papermill records them in the notebook outputs and the docs are still
published with the error visible (nbsphinx_allow_errors in conf.py).
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
            # allow_errors: cell failures (API down, ...) are recorded in the
            # notebook as error outputs instead of aborting execution.
            pm.execute_notebook(
                str(nb), str(nb), kernel_name="python3", allow_errors=True
            )
        except pm.PapermillExecutionError as err:
            # Cell error: papermill already saved the notebook with the error
            # output — publish the docs anyway, so don't fail the run.
            print(
                f"NOTE {nb.name}: cell error recorded ({err.ename}), publishing with error"
            )
        except pm.PapermillException as err:
            # non-cell failures (e.g. kernel won't start) still fail the run
            print(f"FAILED {nb.name}: {type(err).__name__}: {err}", file=sys.stderr)
            status = 1
        print("::endgroup::")
    return status


if __name__ == "__main__":
    sys.exit(main())
