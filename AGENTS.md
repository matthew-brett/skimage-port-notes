- Follow project guidelines on AI authorship; if unspecified, on each
  commit use an `Assisted-by: <harness>:<model>` tag (never `Co-authored-by` or `Signed-off-by`).
- Never open a PR automatically, always offer to make a PR, if a PR is
  something sensible to do for the given task.
- Prefer ASD-STE100 Simplified Technical English
- Unless specifically asked, do not Git-stage changes, and do not make
  commits.

## Code style

- Comments explain the code as it stands, for a reader with zero context on
  how it was written. No references to prior versions, bugs fixed, or changes
  made — including what you just worked out while debugging. If a comment only
  makes sense to someone who watched the code evolve, rewrite or remove it:
  state the lesson as a property or constraint ("end is exclusive"), not as
  history ("this used to be off-by-one").
- Keep comments proportional to the code: usually one line. Don't restate what
  the code says, or justify one line of configuration with a paragraph of
  rationale. If the reasoning needs more than a line or two, it belongs in a
  commit message or README.
- Have a strong preference for code that is easy to read and review.  Optimize
  for clarity and simplicity, and only secondarily for performance, unless otherwise instructed.  Prefer common idioms.
- In making code changes, consider the reviewer.  Prefer to minimize the code
  changes, where possible, and compatible with the rules above.  If the changes are substantial, suggest a plan to break up the code changes into stages that can be implemented as a series of pull-requests, each leaving the code-base in a testable state.

## Python

- Run Python with bare `python` / `python3` from this directory so pyenv
  honours `.python-version`. Do not require or prefer `.venv/bin/python`.

## Output

- Write output files, by default, into the `port-notes` directory.

## Comparative software

- In explanations, review any alternative, well-used implementations in the
  same field, starting with those available in Python.  For example, for image processing, consider Pillow, and OpenCV.
