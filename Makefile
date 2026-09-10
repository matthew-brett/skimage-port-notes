PYTHON ?= python
PIP_INSTALL_CMD ?= $(PYTHON) -m pip install
BUILD_DIR=_build/html
JL_DIR=_build/jl

html:
	# Check for ipynb files in source (should all be .Rmd).
	if compgen -G "*.ipynb" 2> /dev/null; then (echo "ipynb files" && exit 1); fi
	jupyter-book build -W .

github: html
	ghp-import -n _build/html -p -f

clean: rm-ipynb
	rm -rf _build

rm-ipynb:
	rm -rf *.ipynb
