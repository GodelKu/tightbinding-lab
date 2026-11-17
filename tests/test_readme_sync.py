"""README numbers must equal what tools/fill_readme.py builds from the JSON files in docs/."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fill_readme_module():
    spec = importlib.util.spec_from_file_location("fill_readme", ROOT / "tools" / "fill_readme.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_matches_docs_json() -> None:
    mod = _fill_readme_module()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert mod.fill(readme, mod.blocks(mod.load())) == readme, (
        "README is out of date: run bash scripts/make_docs.sh (or python tools/fill_readme.py)"
    )


def test_every_generated_block_has_markers() -> None:
    mod = _fill_readme_module()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in mod.BLOCKS:
        assert f"<!-- BEGIN {name} -->" in readme and f"<!-- END {name} -->" in readme
