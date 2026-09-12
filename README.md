# Port notes for Skimage1/Skimage2

Many of these are AI-generated scripts, or notes.  Therefore, copyright is
ambiguous; please do not re-use, except for those files labeled here, or in
their content, as having specific copyright.

* `coordinate_review.md` is my own (MBs) work; released under CC-By

## Notebook kernel

This tree’s `.python-version` selects the `port-notes` pyenv env for the shell.
Notebook UIs often ignore that and pick a generic `python3` kernel whose
`argv` is bare `python` (whatever is on `PATH`).

Once per machine:

```bash
make kernel
```

That links `.venv` to the pyenv env and registers a user kernelspec named
`port-notes`. Cursor / VS Code then default to `${workspaceFolder}/.venv`
via `.vscode/settings.json`. In the kernel picker, choose
**Python (port-notes)** if a notebook still offers a bare Python 3.
