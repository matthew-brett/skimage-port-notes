# Build, check and publish the port-notes book.
#
# Every target runs Python through the `$(PYTHON)` on PATH, never an absolute
# interpreter path. In this directory `.python-version` makes pyenv resolve
# that to the virtualenv it names, so the notebooks execute in that
# environment: their frontmatter asks for `kernelspec: name: python3`, which
# means "whatever python3 kernel the executing Jupyter offers", so the
# environment follows the caller and the caller is fixed here.
#
# Calling a tool by absolute path (say .../versions/other-env/bin/jupytext)
# silently executes the notebooks somewhere else, with different package
# versions. `make env` shows what is live; the other targets refuse to run
# from the wrong environment.

PYTHON ?= python
PIP_INSTALL_CMD ?= $(PYTHON) -m pip install
BUILD_DIR = _build/html
JL_DIR = _build/jl

NOTEBOOKS := $(shell $(PYTHON) -c "import yaml; d=yaml.safe_load(open('_toc.yml')); \
  print(' '.join(f['file']+'.md' for p in d.get('parts',[]) for f in p['chapters'] \
  if p.get('caption')=='Diagnostic notebooks'))" 2>/dev/null)

.PHONY: help env guard html book check check-all github clean rm-ipynb bresenham-fixtures

help:
	@echo "make env       show the interpreter and kernels the notebooks will use"
	@echo "make check     execute every notebook in the TOC, report errors"
	@echo "make check NB=on_lines.md   execute just one"
	@echo "make html      build the book, warnings as errors"
	@echo "make github    build and publish to GitHub Pages"
	@echo "make clean     remove _build and the paired .ipynb files"
	@echo "make bresenham-fixtures   regenerate bresenham_nd_fixtures/*.json"
	@echo
	@echo "notebooks: $(NOTEBOOKS)"

env:
	@$(PYTHON) -c "import sys, pathlib; \
	print('.python-version :', pathlib.Path('.python-version').read_text().strip() \
	      if pathlib.Path('.python-version').exists() else '(none)'); \
	print('sys.prefix      :', sys.prefix); \
	print('interpreter     :', sys.executable)"
	@$(PYTHON) -m jupyter kernelspec list 2>/dev/null | sed -n '2,$$p' || true

# Refuse to run if $(PYTHON) is not the environment .python-version names.
guard:
	@$(PYTHON) -c "import sys, pathlib; \
	f = pathlib.Path('.python-version'); \
	want = f.read_text().strip() if f.exists() else None; \
	got = pathlib.Path(sys.prefix).name; \
	sys.exit(0) if (want is None or got == want) else \
	  (print(f'refusing: .python-version says {want!r} but this Python is {got!r}.'), \
	   print('  run make from this directory so pyenv resolves the shim,'), \
	   print('  or override, e.g. make PYTHON=/path/to/env/bin/python'), \
	   sys.exit(1))"

html: guard
	# Check for ipynb files in source (should all be paired .md).
	if compgen -G "*.ipynb" 2> /dev/null; then (echo "ipynb files" && exit 1); fi
	$(PYTHON) -c "from jupyter_book.cli.main import main; main()" build -W .

# `book` is an alias for `html`, kept because the notebooks refer to it.
book: html

NB ?=
check: guard
	$(PYTHON) scripts/check_notebooks.py $(NB)

check-all: check

github: html
	ghp-import -n $(BUILD_DIR) -p -f

clean: rm-ipynb
	rm -rf _build

rm-ipynb:
	rm -rf *.ipynb

# Needs the bresenham-nd build (not the port-notes env). Example:
#   PYENV_VERSION=bresenham-nd make bresenham-fixtures
SKIMAGE_ROOT ?= ../bresenham-nd/build-install/usr/lib/python3.13/site-packages
bresenham-fixtures:
	$(PYTHON) bresenham_nd_fixtures/generate_fixtures.py \
	  --skimage-root $(SKIMAGE_ROOT)
