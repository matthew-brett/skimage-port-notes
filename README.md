# Port notes for Skimage1/Skimage2

Many of these are AI-generated scripts, or notes.  Therefore, copyright is
ambiguous; please do not re-use, except for those files labeled here, or in
their content, as having specific copyright.

* `coordinate_review.md` is my own (MBs) work; released under CC-By

## Python environment

`.python-version` names the pyenv virtualenv (`port-notes`). In this directory,
bare `python` / `python3` resolve to that env. Use that; do not require a
`.venv` symlink.

Notebook frontmatter asks for `kernelspec: name: python3` — “whatever
`python3` kernel the executing Jupyter offers” — so the env follows the
caller. `make check` and agent shells that load pyenv already do the right
thing.

Cursor’s workspace setting points at `~/.pyenv/shims/python` so the editor
uses the same shim. An optional `.venv` link is only for tools that insist on
a project-local `bin/python` and will not read pyenv.
