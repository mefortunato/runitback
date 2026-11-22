"""Print the EQUIVALENT_CODE entries (runitback.wager.web.equivalent_code) that map this checkout's
per-sport code hashes to those of older commits.

    python scripts/equivalent_code.py origin/main [more refs...] [--sports nfl cfb ...]

Use it as the last step before a merge, only for sports whose walk-forward results the branch
leaves identical: an entry applies to exactly the source it was computed from, so any later change
to hashed code makes it inert and it must be printed again.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HASHES = (
    "import json; from runitback.wager.core.sport import BUILTIN_SPORTS; "
    "from runitback.wager.web.build import code_hash; "
    "print(json.dumps({s: code_hash(s) for s in BUILTIN_SPORTS}))"
)


def hashes(src: Path) -> dict[str, str]:
    env = {**os.environ, "PYTHONPATH": str(src)}
    out = subprocess.run([sys.executable, "-c", HASHES], env=env, check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("refs", nargs="+", help="git refs whose per-sport hashes the current code is equivalent to")
    parser.add_argument("--sports", nargs="*", help="only these sports (default: those every ref has)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    current = hashes(root / "src")
    older: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, ref in enumerate(args.refs):
            tree = Path(tmp) / f"ref{i}"
            subprocess.run(["git", "-C", str(root), "worktree", "add", "-q", "--detach", str(tree), ref], check=True)
            try:
                for sport, h in hashes(tree / "src").items():
                    if h not in older.setdefault(sport, []):
                        older[sport].append(h)
            finally:
                subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(tree)], check=True)
    for sport in args.sports or [s for s in current if s in older]:
        olds = [h for h in older.get(sport, []) if h != current[sport]]
        if not olds:
            print(f"    # {sport}: unchanged")
            continue
        print(f"    # {sport}\n    \"{current[sport]}\": (")
        for h in olds:
            print(f"        \"{h}\",")
        print("    ),")


if __name__ == "__main__":
    main()
