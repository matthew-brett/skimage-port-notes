#!/usr/bin/env python3
"""Execute the book's notebooks and report cells, figures and errors.

Run through `make check`, which pins the interpreter. Executing this with a
different `python3` executes the notebooks in a different environment: their
frontmatter asks for `kernelspec: name: python3`, which resolves to whatever
Jupyter the caller is running.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import time

import yaml


def toc_notebooks(caption="Diagnostic notebooks"):
    """The `.md` notebooks the table of contents lists under `caption`."""
    toc = yaml.safe_load(pathlib.Path("_toc.yml").read_text())
    return [f"{entry['file']}.md"
            for part in toc.get("parts", [])
            if part.get("caption") == caption
            for entry in part["chapters"]]


def execute(path):
    """Run one notebook; return (code cells, figures, error cells) or None."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "out.ipynb"
        run = subprocess.run(
            [sys.executable, "-m", "jupytext", "--to", "ipynb", "--execute",
             str(path), "-o", str(out)],
            capture_output=True, text=True,
        )
        if run.returncode != 0:
            print(run.stderr[-1200:], file=sys.stderr)
            return None
        cells = json.loads(out.read_text())["cells"]
        code = [c for c in cells if c["cell_type"] == "code"]
        errors = [c for c in code
                  if any(o["output_type"] == "error" for o in c.get("outputs", []))]
        figures = sum(1 for c in code for o in c.get("outputs", [])
                      if "image/png" in o.get("data", {}))
        return code, figures, errors


def main(argv):
    targets = argv[1:] or toc_notebooks()
    print(f"executing with {sys.executable}\n")
    failed = []
    for name in targets:
        started = time.time()
        result = execute(name)
        if result is None:
            print(f"FAIL {name:<24} did not execute")
            failed.append(name)
            continue
        code, figures, errors = result
        print(f"{'FAIL' if errors else 'ok  '} {name:<24}"
              f"{len(code):>4} cells{figures:>4} figures{len(errors):>3} errors"
              f"{time.time() - started:>6.0f}s")
        if errors:
            failed.append(name)
    if failed:
        print(f"\n{len(failed)} notebook(s) with errors: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
