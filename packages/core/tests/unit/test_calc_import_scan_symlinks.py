"""The boundary scanner's walk: anchored skips, importer-shaped skips, no symlinks.

Three findings deferred from the Phase 3B1 review, closed in 3B2 because the
one-door evidence became load-bearing the moment production first imported
the gateway:

* L1 — ``skip_parts`` matched EVERY path component, so a nested
  ``specialists/calc/helper.py`` importing the engine was invisible;
* L2 — ``rglob`` did not descend symlinked directories while the importer
  does, so a linked-in second importer was never walked;
* L3 — attribute traversal and ``sys.modules`` lookups are outside the static
  guard; the resolver's docstring now says so (no detector is added).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.unit._calc_import_scan import (
    SymlinkUnderScanRoot,
    reaches_execution,
    scan_tree,
)

ENGINE_IMPORT = "from openexecutive.calc.engine import execute_batch\n"


def _write(root: Path, relative: str, source: str = ENGINE_IMPORT) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _reaching(root: Path, **kw: object) -> set[str]:
    scanned = scan_tree(root, package_name="openexecutive", **kw)  # type: ignore[arg-type]
    return {
        path.relative_to(root).as_posix()
        for path, targets in scanned.items()
        if reaches_execution(targets)
    }


# --- L1: anchored skips ------------------------------------------------------


def test_skip_parts_apply_to_the_first_component_only(tmp_path: Path) -> None:
    """``calc`` at the root is the package; ``calc`` nested anywhere else is a door."""
    root = tmp_path / "openexecutive"
    _write(root, "calc/engine.py")                     # the package itself: skipped
    _write(root, "specialists/calc/helper.py")         # nested: MUST be scanned
    _write(root, "orchestrator/tests/fixture.py")      # nested `tests`: scanned
    _write(root, "tests/fixture.py")                   # root-level `tests`: skipped

    assert _reaching(root, skip_parts=("calc", "tests")) == {
        "specialists/calc/helper.py",
        "orchestrator/tests/fixture.py",
    }


def test_a_skipped_name_at_the_root_is_not_skipped_as_a_file(tmp_path: Path) -> None:
    """``root/calc.py`` is a module, not the skipped directory."""
    root = tmp_path / "openexecutive"
    _write(root, "calc.py")
    assert _reaching(root, skip_parts=("calc",)) == {"calc.py"}


# --- importer-shaped skips ---------------------------------------------------


def test_only_pycache_is_skipped_by_name_at_any_depth(tmp_path: Path) -> None:
    """Non-identifier directory names are importable via importlib and are scanned.

    Security review of the first version found the opposite premise false:
    ``importlib.import_module("pkg.calc-helpers.door")`` works, so a scanner
    that skipped ``calc-helpers`` hid a second door.
    """
    root = tmp_path / "openexecutive"
    _write(root, "orchestrator/door.py")
    _write(root, "orchestrator/__pycache__/stale.py")     # no source lives here
    _write(root, "calc-helpers/door.py")                  # importable; scanned
    _write(root, "calc.v2/door.py")                       # importable; scanned
    _write(root, "orchestrator/.hidden/door.py")          # importable; scanned
    _write(root, "node_modules/x/y.py")
    assert _reaching(root) == {
        "orchestrator/door.py", "calc-helpers/door.py", "calc.v2/door.py",
        "orchestrator/.hidden/door.py", "node_modules/x/y.py",
    }


def test_a_dot_venv_is_skipped_only_when_the_caller_anchors_it(tmp_path: Path) -> None:
    """``.venv`` is not special: it is scanned unless named in skip_parts at the root."""
    root = tmp_path / "openexecutive"
    _write(root, "orchestrator/door.py")
    _write(root, ".venv/lib/site.py")
    assert _reaching(root) == {"orchestrator/door.py", ".venv/lib/site.py"}
    assert _reaching(root, skip_parts=(".venv",)) == {"orchestrator/door.py"}


def test_an_ancestor_directory_name_cannot_silence_the_scan(tmp_path: Path) -> None:
    root = tmp_path / "tests" / "calc" / ".venv-like" / "openexecutive"
    _write(root, "orchestrator/door.py")
    assert _reaching(root, skip_parts=("calc", "tests")) == {"orchestrator/door.py"}


# --- L2: symlinks are rejected, never followed -------------------------------


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform has no symlinks")
def test_a_symlinked_directory_under_the_root_fails_the_scan(tmp_path: Path) -> None:
    root = tmp_path / "openexecutive"
    _write(root, "orchestrator/door.py")
    elsewhere = tmp_path / "elsewhere"
    _write(elsewhere, "hidden.py")
    (root / "linked").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(SymlinkUnderScanRoot) as info:
        scan_tree(root, package_name="openexecutive")
    assert info.value.path == root / "linked"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform has no symlinks")
def test_a_symlinked_file_under_the_root_fails_the_scan(tmp_path: Path) -> None:
    root = tmp_path / "openexecutive"
    real = _write(tmp_path / "elsewhere", "hidden.py")
    (root / "orchestrator").mkdir(parents=True)
    (root / "orchestrator" / "alias.py").symlink_to(real)

    with pytest.raises(SymlinkUnderScanRoot):
        scan_tree(root, package_name="openexecutive")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform has no symlinks")
def test_a_symlink_inside_a_skipped_directory_still_fails(tmp_path: Path) -> None:
    """Checked before any skip rule, so no directory name can hide one."""
    root = tmp_path / "openexecutive"
    _write(root, "orchestrator/door.py")
    (root / "calc").mkdir()
    (root / "calc" / "escape").symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(SymlinkUnderScanRoot):
        scan_tree(root, package_name="openexecutive", skip_parts=("calc",))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform has no symlinks")
def test_a_symlinked_root_fails_the_scan(tmp_path: Path) -> None:
    real = tmp_path / "real"
    _write(real, "orchestrator/door.py")
    link = tmp_path / "openexecutive"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SymlinkUnderScanRoot) as info:
        scan_tree(link, package_name="openexecutive")
    assert info.value.path == link


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform has no symlinks")
def test_a_symlink_cycle_cannot_hang_the_walk(tmp_path: Path) -> None:
    """Rejection is what keeps the walk bounded: a cycle is refused, not followed."""
    root = tmp_path / "openexecutive"
    _write(root, "orchestrator/door.py")
    (root / "orchestrator" / "loop").symlink_to(root, target_is_directory=True)
    with pytest.raises(SymlinkUnderScanRoot):
        scan_tree(root, package_name="openexecutive")


def test_the_real_production_tree_has_no_symlinks() -> None:
    """The scan that backs the one-door invariant must be able to run at all."""
    root = Path(__file__).resolve().parents[2] / "openexecutive"
    scan_tree(root, package_name="openexecutive", skip_parts=("calc",))


# --- L3: scope is stated, not detected --------------------------------------


def test_attribute_traversal_is_named_as_out_of_scope() -> None:
    import tests.unit._calc_import_scan as resolver

    doc = resolver.__doc__ or ""
    for phrase in ("Attribute traversal", "sys.modules", "getattr", "controlled by review"):
        assert phrase in doc, phrase
