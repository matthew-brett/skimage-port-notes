#!/usr/bin/env python3
"""Post a Jupyter notebook as a GitHub issue comment with plot images.

Copies ``name.ipynb`` into ``name/``, converts it to Markdown, then runs
``gh issue comment`` with ``--attach`` for each local image so GitHub hosts
the figures and rewrites the Markdown links.  Deletes ``name/`` afterward
unless ``--retain`` is set.

The GitHub repository is ``--repo`` if given, else the nearest ``.default-repo``
file (this path, then parents), else the nearest git checkout that ``gh`` can
resolve.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'}
_MD_IMAGE_RE = re.compile(
    r'''
    !\[[^\]]*\]          # ![alt]
    \(\s*
    <?([^)\s>]+)>?       # url / path
    (?:\s+["'][^"']*["'])?
    \s*\)
    ''',
    flags=re.VERBOSE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Convert a notebook to Markdown and post it as a GitHub issue '
            'comment with attached images.'
        )
    )
    parser.add_argument('issue', type=int, help='GitHub issue number')
    parser.add_argument(
        'notebook',
        type=Path,
        help='Path to the .ipynb file',
    )
    parser.add_argument(
        '--repo',
        help=(
            'owner/name (default: nearest .default-repo file, else nearest '
            'GitHub repo from the notebook path)'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Convert and print the gh command; do not post',
    )
    parser.add_argument(
        '--retain',
        action='store_true',
        help='Keep the generated notebook directory (default: delete it)',
    )
    args = parser.parse_args(argv)

    notebook = args.notebook.expanduser().resolve()
    if not notebook.is_file():
        print(f'error: notebook not found: {notebook}', file=sys.stderr)
        return 1
    if notebook.suffix.lower() != '.ipynb':
        print(f'error: expected an .ipynb file, got: {notebook}', file=sys.stderr)
        return 1

    out_dir = notebook.parent / notebook.stem
    try:
        return _convert_and_post(args, notebook, out_dir)
    finally:
        if not args.retain and out_dir.exists():
            shutil.rmtree(out_dir)


def _convert_and_post(args: argparse.Namespace, notebook: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_nb = out_dir / notebook.name
    shutil.copy2(notebook, dest_nb)

    _run(
        [
            sys.executable,
            '-m',
            'jupyter',
            'nbconvert',
            '--to',
            'markdown',
            str(dest_nb),
        ],
        cwd=out_dir,
    )

    md_path = out_dir / f'{notebook.stem}.md'
    if not md_path.is_file():
        print(f'error: markdown not created: {md_path}', file=sys.stderr)
        return 1

    images = _collect_images(md_path, out_dir)
    repo = args.repo or _resolve_repo(notebook.parent)

    cmd = [
        'gh',
        'issue',
        'comment',
        str(args.issue),
        '--repo',
        repo,
        '--body-file',
        md_path.name,
    ]
    for image in images:
        cmd.extend(['--attach', str(image)])

    if args.dry_run:
        print('cwd:', out_dir)
        print('repo:', repo)
        print('images:', ', '.join(map(str, images)) or '(none)')
        print('command:', subprocess.list2cmdline(cmd))
        return 0

    _run(cmd, cwd=out_dir)
    print(f'Posted comment on {repo}#{args.issue} from {md_path}')
    return 0


def _collect_images(md_path: Path, out_dir: Path) -> list[Path]:
    """Return image paths relative to `out_dir`, Markdown order then extras."""
    text = md_path.read_text(encoding='utf-8')
    found: list[Path] = []
    seen: set[Path] = set()

    def add(rel: Path) -> None:
        rel = Path(rel.as_posix())
        if rel in seen:
            return
        full = (out_dir / rel).resolve()
        try:
            full.relative_to(out_dir.resolve())
        except ValueError:
            return
        if full.is_file() and full.suffix.lower() in _IMAGE_EXTS:
            seen.add(rel)
            found.append(rel)

    for match in _MD_IMAGE_RE.finditer(text):
        raw = match.group(1).strip().strip('<>')
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', raw):
            continue  # http(s), data:, etc.
        add(Path(raw))

    files_dir = out_dir / f'{md_path.stem}_files'
    if files_dir.is_dir():
        for path in sorted(files_dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                add(path.relative_to(out_dir))

    return found


def _resolve_repo(start: Path) -> str:
    """Return owner/name from ``.default-repo`` or the nearest GitHub checkout."""
    default = _nearest_default_repo(start)
    if default:
        return default
    return _nearest_github_repo(start)


def _nearest_default_repo(start: Path) -> str | None:
    """Return ``owner/name`` from the nearest ``.default-repo`` file, if any."""
    start = start.resolve()
    for directory in [start, *start.parents]:
        path = directory / '.default-repo'
        if not path.is_file():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if not re.fullmatch(r'[^/\s]+/[^/\s]+', line):
                print(
                    f'error: invalid repository in {path}: {line!r} '
                    f'(expected owner/name)',
                    file=sys.stderr,
                )
                sys.exit(1)
            return line
        print(f'error: empty .default-repo file: {path}', file=sys.stderr)
        sys.exit(1)
    return None


def _nearest_github_repo(start: Path) -> str:
    start = start.resolve()
    candidates = [start, *start.parents]
    for directory in candidates:
        if not (directory / '.git').exists():
            continue
        result = subprocess.run(
            [
                'gh',
                'repo',
                'view',
                '--json',
                'nameWithOwner',
                '--jq',
                '.nameWithOwner',
            ],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        name = result.stdout.strip()
        if result.returncode == 0 and name:
            return name
        remote = _owner_repo_from_git_remote(directory)
        if remote:
            return remote
    print(
        'error: no GitHub repository found from '
        f'{start} or its parents (need git + gh)',
        file=sys.stderr,
    )
    sys.exit(1)


def _owner_repo_from_git_remote(directory: Path) -> str | None:
    result = subprocess.run(
        ['git', 'remote', 'get-url', 'origin'],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    # git@github.com:owner/repo.git  or  https://github.com/owner/repo.git
    match = re.search(r'github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$', url)
    if not match:
        return None
    return f'{match.group("owner")}/{match.group("repo")}'


def _run(cmd: list[str], *, cwd: Path) -> None:
    print('+', subprocess.list2cmdline(cmd), file=sys.stderr)
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        print(f'error: command not found: {cmd[0]}', file=sys.stderr)
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == '__main__':
    raise SystemExit(main())
