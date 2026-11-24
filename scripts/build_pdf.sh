#!/bin/bash
# Build a PDF from a jupytext notebook (.md or .ipynb).
#
# Uses port-notes/nbconvert_templates/unicode_latex so XeLaTeX gets
# Unicode fonts and pandoc's captionless-longtable \newcounter{none}
# (see nbconvert_templates/README.md).
#
# An unquoted YAML date is a datetime.date. The ipynb writer stores
# metadata as JSON, so each date is written as an ISO string.

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Specify notebook file." >&2
    exit 1
fi
nb_file=$1
if [ ! -f "$nb_file" ]; then
    echo "No such file: $nb_file" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "$0")" && pwd)
templates_dir="${script_dir}/../nbconvert_templates"

python - "$nb_file" << 'PY'
import datetime
import os
import sys

import jupytext
from nbconvert.preprocessors import ExecutePreprocessor


def stringify_dates(value):
    """Replace each date in a notebook tree with an ISO string."""
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, dict):
        for key in list(value):
            value[key] = stringify_dates(value[key])
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = stringify_dates(item)
        return value
    return value


nb_file = sys.argv[1]
notebook = stringify_dates(jupytext.read(nb_file))
kernel_name = notebook.metadata.get("kernelspec", {}).get("name")
print(f"[build_pdf] Executing notebook with kernel {kernel_name}", flush=True)
nb_dir = os.path.dirname(nb_file) or os.getcwd()
ExecutePreprocessor(timeout=None, kernel_name=kernel_name).preprocess(
    notebook, {"metadata": {"path": nb_dir}}
)
dest = os.path.splitext(nb_file)[0] + ".ipynb"
print(f"[build_pdf] Writing {dest}", flush=True)
jupytext.write(notebook, dest, fmt="ipynb")
# The text file must be at least as new as the paired ipynb.
if not nb_file.endswith(".ipynb"):
    os.utime(nb_file, None)
PY
nb_base="${nb_file%.*}"
jupyter nbconvert "${nb_base}.ipynb" --to pdf \
  --template unicode_latex \
  --TemplateExporter.extra_template_basedirs="${templates_dir}"
