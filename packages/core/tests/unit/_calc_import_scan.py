"""One conservative AST import resolver, shared by every calc boundary scanner.

Three tests enforce the calc boundary — one bounds who may import the package
at all (``test_calc_contract_foundation``), one bounds who may execute a
calculation (``test_calc_adversarial``), and the gateway suite re-asserts the
one-door property from its own side (``test_calculation_gateway``). They once
had *separate* copies of the resolution logic, each reading only
``ImportFrom.module`` and ignoring ``ImportFrom.names``, so five ordinary import
forms slipped past one or more of them:

    from openexecutive import calc          MISSED by all three
    from openexecutive.calc import engine   MISSED by both engine scanners
    from openexecutive.calc import authority        "
    from .. import calc                             "
    from ..calc import engine                       "

The engine scanners missed four of them because they matched only the dotted
string ``openexecutive.calc.engine``; ``from openexecutive.calc import engine``
names the module in an *alias*, not in ``node.module``.

**Why binding the bare package counts as reaching the engine.**
``calc/__init__`` re-exports ``execute``, ``execute_batch``,
``issue_calculation_result`` and ``current_authority``. So a module holding the
package object — ``from openexecutive import calc`` — can call
``calc.execute_batch(...)`` without ever naming the engine module. Being allowed
to import calc *types* must not silently confer a second route to execute, so
:func:`reaches_execution` treats a package binding as execution-reaching while
leaving ``from openexecutive.calc import Unit`` alone: that binds a type, not
the package.

**Dynamic imports — what this guard does and does not see.**
A call to ``importlib.import_module`` or ``__import__`` whose module name is a
string *literal* is resolved like an import statement (see
:func:`_literal_dynamic_import`). That is the whole of the dynamic-import
coverage. A name computed at runtime, an aliased ``importlib`` binding, a
relative literal such as ``import_module("..calc", package=__package__)``, or a
module reached through a dependency are **not statically decidable here** and
are outside this guard; they are controlled by review, not by this scan. The
tests state that scope rather than let a green run imply more than it checked.

**Filtering is anchored, and almost nothing is skipped by rule.** ``scan_tree``
skips two kinds of directory and nothing else:

* at ANY depth, ``__pycache__`` — it holds no ``.py`` source, so there is
  nothing to parse. That is the ONLY name-based rule. A draft also skipped
  directories whose name is not an identifier (``.venv``, ``calc-helpers``) on
  the premise that the importer cannot enter them; a probe showed the premise
  is false — ``importlib.import_module("pkg.calc-helpers.door")`` imports
  fine, and this module already models that literal-string form as a reach.
  So such directories are scanned like any other;
* by ANCHORED root-relative path only, the caller's ``skip_parts`` (``calc``,
  ``tests``, ``.venv``): matched against the FIRST component below ``root``. An earlier
  version matched every component, so a future
  ``openexecutive/specialists/calc/helper.py`` importing the engine was
  invisible to all three scanners; and the version before that matched the
  absolute path, so a scan root that merely lived under a directory named
  ``tests`` (a CI checkout, a ``tmp_path``) scanned zero files and the boundary
  passed vacuously.

**Symlinks are rejected, not followed.** The walk is ``os.walk`` with the
default ``followlinks=False``, and any symlinked directory OR file met under
the root raises :class:`SymlinkUnderScanRoot` naming it — checked before any
skip rule, so no directory name can silence it. A scan that meets a symlink
cannot make the one-door claim: the importer follows links and this walk does
not, so a linked-in second importer would be unscanned. Rejecting rather than
following is what keeps the walk bounded — it cannot leave the root, cannot
cycle, and cannot read one real file twice — and it matches the calc-internal
walk in ``test_calc_contract_foundation``. A repository that needs a symlink
under a scanned root widens this deliberately, with the test failing until it
does.

**Reach forms this static guard does not see — stated, not detected.**
Attribute traversal after a permitted import (``import openexecutive`` then
``openexecutive.calc.execute_batch``), ``sys.modules["openexecutive.calc"]``,
``getattr(pkg, "execute_batch")`` and ``globals()``/``vars()`` lookups all
reach the engine without an import statement naming it, and none is decidable
from an import table. They are controlled by review — the same status as the
computed dynamic-import forms below — and no detector is added for them here:
an attribute-traversal detector is a false-positive generator, and unreviewed
logic on the evidence path is worse than an honest scope statement.

The resolver is deliberately conservative. It over-reports rather than
under-reports (``from A import B`` is recorded as exposing ``A.B`` whether or not
``B`` is a submodule), because a false positive here is a visible test failure
someone fixes, while a false negative is a silent second door.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

CALC_PACKAGE = "openexecutive.calc"

EXECUTION_MODULES = frozenset(
    {
        "openexecutive.calc.engine",
        # The authority minter belongs here for the same reason the engine does:
        # ``issue_calculation_result`` is what stamps a record as coming from the
        # application. A second caller of it is a second place records can be
        # minted, which the one-door invariant exists to make countable.
        "openexecutive.calc.authority",
    }
)
"""Modules whose import is, by itself, a route to execute or to mint authority."""

EXECUTION_NAMES = frozenset(
    {"execute", "execute_batch", "issue_calculation_result", "current_authority"}
)
"""Names re-exported by ``calc/__init__`` that execute or mint authority.

``from openexecutive.calc import execute_batch`` reaches the engine without
naming ``engine``, so the alias has to be inspected, not just the module.
"""


def _package_parts(path: Path, root: Path, package_name: str | None) -> list[str]:
    """The dotted package a file lives in, for resolving its relative imports.

    ``package_name`` is ``None`` for scanned trees that are not importable
    packages (``scripts/``, ``evals/``); a relative import there cannot resolve
    to ``openexecutive.calc``, so the caller gets an empty base and the import is
    simply not attributed to calc.
    """
    if package_name is None:
        return []
    return [package_name, *path.relative_to(root).parts[:-1]]


def _literal_dynamic_import(node: ast.Call) -> tuple[str, bool] | None:
    """``(module_name, binds_prefixes)`` for a literal dynamic import, else None.

    Recognised spellings, and only these:

    * ``importlib.import_module("a.b.c")`` — returns the *leaf* module object,
      so only ``a.b.c`` is recorded as bound.
    * ``import_module("a.b.c")`` — the bare name after
      ``from importlib import import_module``; same semantics.
    * ``__import__("a.b.c")`` — returns the *top-level* package, so every
      prefix is reachable through it, exactly as for ``import a.b.c``.

    Anything else — a computed name, an aliased ``importlib``, a relative
    literal (leading dot; its ``package=`` argument is almost never a literal),
    a ``Constant`` that is not a ``str`` — returns ``None`` and is, by design,
    not this guard's concern.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        is_import_module = (
            isinstance(func.value, ast.Name)
            and func.value.id == "importlib"
            and func.attr == "import_module"
        )
        binds_prefixes = False
        if not is_import_module:
            return None
    elif isinstance(func, ast.Name) and func.id in ("import_module", "__import__"):
        binds_prefixes = func.id == "__import__"
    else:
        return None

    if not node.args:
        return None
    first = node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    name = first.value
    if not name or name.startswith("."):
        return None
    return name, binds_prefixes


def import_targets(
    tree: ast.AST, *, path: Path, root: Path, package_name: str | None
) -> set[tuple[str, bool]]:
    """Every module or attribute an import exposes to ``path``.

    Returns ``(dotted_target, binds_target)`` pairs. ``binds_target`` is ``True``
    when the statement puts *that* name in the module's namespace — which is what
    distinguishes ``from openexecutive import calc`` (binds the package, so
    ``calc.execute_batch`` is reachable) from ``from openexecutive.calc import
    Unit`` (binds a type; the package object is never held).

    Import statements are resolved in full, relative levels included. Dynamic
    imports are resolved only when the module name is a string literal in one
    of the spellings :func:`_literal_dynamic_import` lists.
    """
    targets: set[tuple[str, bool]] = set()
    package = _package_parts(path, root, package_name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # ``import a.b.c`` makes every prefix reachable through the binding.
            for alias in node.names:
                parts = alias.name.split(".")
                for index in range(1, len(parts) + 1):
                    targets.add((".".join(parts[:index]), True))
            continue

        if isinstance(node, ast.Call):
            literal = _literal_dynamic_import(node)
            if literal is None:
                continue
            name, binds_prefixes = literal
            parts = name.split(".")
            if binds_prefixes:
                for index in range(1, len(parts) + 1):
                    targets.add((".".join(parts[:index]), True))
            else:
                targets.add((name, True))
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        level = node.level or 0
        if level == 0:
            base = node.module.split(".") if node.module else []
        else:
            # ``level`` 1 is the current package, 2 its parent, and so on.
            upto = len(package) - (level - 1)
            if upto < 1:
                continue  # escapes the scanned tree; nothing to attribute
            base = package[:upto]
            if node.module:
                base = [*base, *node.module.split(".")]
        if not base:
            continue

        # The source module is referenced but not bound: ``from X import Y``
        # leaves ``X`` out of the namespace.
        targets.add((".".join(base), False))
        for alias in node.names:
            if alias.name == "*":
                # A star import exposes whatever the module re-exports, which
                # for ``calc`` includes the execution surface. Recorded as a
                # binding of the module itself.
                targets.add((".".join(base), True))
                continue
            targets.add((".".join([*base, alias.name]), True))

    return targets


def references_calc(targets: set[tuple[str, bool]]) -> set[str]:
    """Targets that touch ``openexecutive.calc`` in any way."""
    return {
        target
        for target, _binds in targets
        if target == CALC_PACKAGE or target.startswith(f"{CALC_PACKAGE}.")
    }


def reaches_execution(targets: set[tuple[str, bool]]) -> set[str]:
    """Targets that confer the ability to execute a calculation or mint a stamp.

    Three shapes, and the third is the one the old scanners missed entirely:

    * an execution module, however it was named;
    * an execution name imported from the package;
    * a **binding of the package itself**, because ``calc/__init__`` re-exports
      the execution surface.
    """
    execution_names = {f"{CALC_PACKAGE}.{name}" for name in EXECUTION_NAMES}
    reaching: set[str] = set()
    for target, binds in targets:
        names_execution_module = any(
            target == module or target.startswith(f"{module}.")
            for module in EXECUTION_MODULES
        )
        if (
            names_execution_module
            or target in execution_names
            or (binds and target == CALC_PACKAGE)
        ):
            reaching.add(target)
    return reaching


class SymlinkUnderScanRoot(Exception):
    """A symlink was met under a scanned root; the one-door claim is void."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"symlink under scan root: {path}")
        self.path = path


def _unimportable_dir(name: str) -> bool:
    """A directory that can hold no ``.py`` source. Only ``__pycache__``.

    Deliberately NOT ``not name.isidentifier()``: ``importlib.import_module``
    does not validate identifiers, so a directory named ``calc-helpers`` is
    importable through the literal dynamic form this module already treats as
    a reach. Skipping it would hide a second door.
    """
    return name == "__pycache__"


def _source_files(root: Path) -> list[Path]:
    """Every real ``.py`` file under ``root``, refusing any symlink on the way.

    The root itself is checked too: a scan pointed at a symlink would walk the
    link target and report paths under a root that is not where the files are.
    """
    if root.is_symlink():
        raise SymlinkUnderScanRoot(root)
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        for name in dirnames:
            if (here / name).is_symlink():
                raise SymlinkUnderScanRoot(here / name)
        dirnames[:] = [name for name in dirnames if not _unimportable_dir(name)]
        for name in filenames:
            path = here / name
            if path.is_symlink():
                raise SymlinkUnderScanRoot(path)
            if path.suffix == ".py":
                files.append(path)
    return sorted(files)


def scan_tree(
    root: Path, *, package_name: str | None, skip_parts: tuple[str, ...] = ()
) -> dict[Path, set[tuple[str, bool]]]:
    """Resolve the import targets of every ``.py`` file under ``root``.

    ``skip_parts`` is matched against the FIRST path component *below* ``root``
    only — ``("calc",)`` skips ``root/calc/…`` and nothing else, so a nested
    ``root/specialists/calc/…`` is scanned. ``__pycache__`` is skipped at any
    depth; nothing else is skipped by name. A symlink anywhere under ``root``
    (or ``root`` itself) raises rather than being followed or ignored; see the
    module docstring.
    """
    found: dict[Path, set[tuple[str, bool]]] = {}
    for path in _source_files(root):
        relative = path.relative_to(root).parts
        if len(relative) > 1 and relative[0] in skip_parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:  # pragma: no cover - not our code to fix
            continue
        found[path] = import_targets(
            tree, path=path, root=root, package_name=package_name
        )
    return found
