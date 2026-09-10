#!/usr/bin/env python3
"""Create a GitHub issue from a Markdown file.

The title comes from the YAML front matter ``title`` key or, failing that, from
the first top-level ``#`` heading, which is then dropped from the body.

Front matter keys follow GitHub's issue template schema, so the same file also
works under ``.github/ISSUE_TEMPLATE``::

    ---
    labels: [enhancement, ":cyclone: transform"]
    assignees: matthew-brett
    ---

    # Flip the coordinate convention in warp

    Body text.

Unrecognized arguments go to ``gh issue create``, so ``--repo``, ``--parent``
and the rest work as usual.
"""

import re
import shlex
import subprocess
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import yaml

_FRONT_MATTER = re.compile(r'\A---[ \t]*\n(.*?\n)?---[ \t]*\n', re.DOTALL)
_H1 = re.compile(r'^#[ \t]+(.+?)[ \t]*$', re.MULTILINE)

# Front matter keys that map to a repeatable `gh` flag.
_MULTI = {'labels': '--label', 'assignees': '--assignee', 'projects': '--project'}
# Front matter keys that map to a flag given once.
_SINGLE = {'milestone': '--milestone', 'type': '--type'}


def parse(text):
    """Return (title, body, metadata) for Markdown `text`."""
    match = _FRONT_MATTER.match(text)
    meta = (yaml.safe_load(match.group(1) or '') or {}) if match else {}
    body = text[match.end() :] if match else text
    title = meta.get('title')
    if title is None:
        heading = _H1.search(body)
        if heading is None:
            raise ValueError('no `title` in front matter and no `# ` heading')
        title = heading.group(1)
        # GitHub shows the title above the body, so drop the duplicate heading.
        body = body[: heading.start()] + body[heading.end() + 1 :]
    return str(title), body.strip() + '\n', meta


def as_list(value):
    """Return `value` as a list, splitting a comma-separated string."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(',') if v.strip()]
    return [str(v) for v in value]


def gh_command(title, meta, extra):
    """Return the ``gh issue create`` command line, reading the body from stdin."""
    cmd = ['gh', 'issue', 'create', '--title', title, '--body-file', '-']
    for key, flag in _MULTI.items():
        for value in as_list(meta.get(key)):
            cmd += [flag, value]
    for key, flag in _SINGLE.items():
        if meta.get(key) is not None:
            cmd += [flag, str(meta[key])]
    return cmd + extra


def main():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument('markdown', help='Markdown file describing the issue')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print the command and body, create nothing',
    )
    args, extra = parser.parse_known_args()
    title, body, meta = parse(Path(args.markdown).read_text())
    cmd = gh_command(title, meta, extra)
    if args.dry_run:
        print(shlex.join(cmd), end='\n\n')
        print(body)
        return 0
    return subprocess.run(cmd, input=body, text=True).returncode


if __name__ == '__main__':
    sys.exit(main())
