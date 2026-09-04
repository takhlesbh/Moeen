"""The trust boundary: what it enforces, and — stated honestly — what it cannot."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from openexecutive.evidence import contracts
from openexecutive.evidence._authority import (
    UntrustedConstructionError,
    require_trusted_construction,
    trusted_construction,
)
from openexecutive.evidence.contracts import DocumentVersion, Extraction, SourceSpan

PACKAGE_ROOT = Path(contracts.__file__).resolve().parents[1]
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence"
TRUSTED_NAME = "trusted_construction"
ALLOWED_REFERENCES = {"_authority.py", "factory.py"}
HEX = "a" * 64

CANONICAL_PAYLOADS: dict[Any, dict[str, Any]] = {
    DocumentVersion: {
        "document_version_id": HEX, "scope_id": "acme", "logical_source_id": HEX,
        "content_sha256": HEX, "byte_size": 1,
    },
    Extraction: {
        "extraction_id": HEX, "document_version_id": HEX, "extractor_name": "pdftext",
        "extractor_version": "1.0.0", "extractor_config_sha256": HEX,
        "raw_text_sha256": HEX, "nfc_text_sha256": HEX, "nfc_text_length": 3,
    },
    SourceSpan: {
        "span_id": HEX, "extraction_id": HEX, "nfc_text_sha256": HEX,
        "start_char": 0, "end_char": 3, "quote": "abc", "quote_sha256": HEX,
    },
}


@pytest.mark.parametrize("model", list(CANONICAL_PAYLOADS))
def test_canonical_model_validate_fails_outside_the_trusted_context(model: Any) -> None:
    with pytest.raises(UntrustedConstructionError):
        model.model_validate(CANONICAL_PAYLOADS[model])


@pytest.mark.parametrize("model", list(CANONICAL_PAYLOADS))
def test_canonical_model_validate_json_fails_outside_the_trusted_context(model: Any) -> None:
    with pytest.raises(UntrustedConstructionError):
        model.model_validate_json(json.dumps(CANONICAL_PAYLOADS[model]))


@pytest.mark.parametrize("model", list(CANONICAL_PAYLOADS))
def test_canonical_construction_succeeds_inside_the_trusted_context(model: Any) -> None:
    """Documents the honest scope: any Python caller, tests included, may enter."""
    with trusted_construction():
        assert model.model_validate(CANONICAL_PAYLOADS[model])


@pytest.mark.parametrize("model", list(CANONICAL_PAYLOADS))
def test_the_trusted_flag_is_not_a_model_field(model: Any) -> None:
    assert TRUSTED_NAME not in model.model_fields
    payload = {**CANONICAL_PAYLOADS[model], TRUSTED_NAME: True}
    with pytest.raises(UntrustedConstructionError):
        model.model_validate_json(json.dumps(payload))


def test_the_context_does_not_leak_after_the_block() -> None:
    with trusted_construction():
        require_trusted_construction("DocumentVersion")
    with pytest.raises(UntrustedConstructionError):
        require_trusted_construction("DocumentVersion")


def test_the_context_is_restored_after_an_exception_inside_the_block() -> None:
    with pytest.raises(RuntimeError), trusted_construction():
        raise RuntimeError("boom")
    with pytest.raises(UntrustedConstructionError):
        require_trusted_construction("DocumentVersion")


def test_nested_contexts_restore_the_outer_state() -> None:
    with trusted_construction():
        with trusted_construction():
            require_trusted_construction("Extraction")
        require_trusted_construction("Extraction")


def test_the_error_names_the_model_and_the_sanctioned_path() -> None:
    with pytest.raises(UntrustedConstructionError) as excinfo:
        DocumentVersion.model_validate(CANONICAL_PAYLOADS[DocumentVersion])
    message = str(excinfo.value)
    assert "DocumentVersion" in message
    assert "openexecutive.evidence.factory" in message


def _modules_referencing(name: str, root: Path) -> set[Path]:
    hits: set[Path] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                (isinstance(node, ast.Name) and node.id == name)
                or (isinstance(node, ast.Attribute) and node.attr == name)
                or (isinstance(node, ast.alias) and name in (node.name, node.asname))
                or (isinstance(node, ast.FunctionDef) and node.name == name)
            ):
                hits.add(path)
    return hits


def test_only_the_factory_references_trusted_construction_in_production() -> None:
    hits = _modules_referencing(TRUSTED_NAME, PACKAGE_ROOT)
    assert {p.name for p in hits} == ALLOWED_REFERENCES
    assert EVIDENCE_ROOT / "factory.py" in hits


def test_evidence_is_a_leaf_package() -> None:
    """It may reach only stdlib, pydantic and itself."""
    forbidden = set()
    for path in EVIDENCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                parts = module.split(".")
                if parts[0] == "openexecutive" and parts[1:2] != ["evidence"]:
                    forbidden.add((path.name, module))
    assert forbidden == set()
