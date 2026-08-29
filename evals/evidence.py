"""Immutable, auditable evidence for one eval run.

The previous harness wrote a flat JSON list holding only the Executive's final
text and the parsed judge scores. A baseline captured that way cannot be
audited later: it does not record which models answered, which specialists were
consulted, what the judge actually said before parsing, how long anything took,
or what knowledge the run was graded against.

Design rules here:

* One directory per run, keyed by a timestamp + random suffix. A run never
  overwrites another — reproducing a regression means comparing two runs, which
  is impossible if the second clobbers the first.
* The manifest is written BEFORE execution with ``status: INCOMPLETE``. A
  killed process therefore leaves visible evidence that it was killed, instead
  of leaving no trace and looking like it was never started.
* ``COMPLETE`` is only ever written after every discovered scenario executed
  and no infrastructure error remains.
* Fields that are genuinely not observable are persisted as
  ``{"status": "unavailable", "reason": ..., "seam": ...}`` — never omitted and
  never faked. A missing key reads as "nobody looked"; this reads as "here is
  exactly why it cannot be known".
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_COMPLETE = "COMPLETE"
STATUS_FAILED = "FAILED"

# Per-scenario outcomes. ERROR is deliberately distinct from FAIL: a model that
# answered badly and a harness that could not obtain a verdict are different
# facts, and collapsing them is how a broken judge used to look like a bad
# score.
OUTCOME_PASS = "PASS"
OUTCOME_FAIL = "FAIL"
OUTCOME_ERROR = "ERROR"

# Random suffix on the run id. Eight hex chars is ~4 billion values, so two
# runs started in the same second effectively never collide — and a collision
# raises rather than overwriting (see RunEvidence.__init__).
RUN_ID_SUFFIX_LEN = 8


# Characters permitted in a scenario id when it is used as a FILENAME. The id
# reaches us from scenario YAML, which can come from an operator-supplied
# --scenarios directory or a shared scenario pack — neither is validated by the
# packaged loader. Without this, `id: ../../../x` writes outside the run
# directory and can overwrite any .json the process can reach.
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]")
# Bound the filename so a very long id cannot trip ENAMETOOLONG mid-run.
_MAX_ID_LEN = 100


def safe_filename(scenario_id: str) -> str:
    """A scenario id reduced to something safe AND unique as a filename.

    Allowlist, not blocklist: every character outside ``[A-Za-z0-9._-]``
    becomes ``_``, and leading dots are stripped so ``..`` cannot survive.

    Sanitizing alone is many-to-one — ``finance/001``, ``finance:001`` and the
    homoglyph ``finance\uff3f001`` all collapse to ``finance_001``, and ids
    differing only past the length cap collide too. Silently overwriting one
    scenario's evidence with another's while the manifest still counts both is
    exactly the kind of untrue bookkeeping this module exists to prevent, so a
    short digest of the ORIGINAL id disambiguates whenever sanitizing changed
    anything. Ids that are already safe (every real scenario) keep their plain
    readable name.
    """
    cleaned = _SAFE_ID.sub("_", scenario_id).lstrip(".")[:_MAX_ID_LEN]
    if cleaned == scenario_id:
        return cleaned
    digest = hashlib.sha256(scenario_id.encode()).hexdigest()[:8]
    return f"{cleaned or 'unnamed'}-{digest}"


def _atomic_write(path: Path, payload: str) -> None:
    """Write via a temp file + ``os.replace`` in the same directory.

    Same reason the manifest does it: a kill mid-write must not leave a
    truncated JSON file that later reads as corrupt evidence.
    """
    tmp = path.with_name(path.name + ".tmp")
    # Drop any stale sidecar first: a leftover file from a crash, or a symlink
    # planted at the predictable .tmp name, which a plain write_text would
    # follow and truncate.
    with contextlib.suppress(FileNotFoundError):
        os.unlink(tmp)
    # O_EXCL|O_NOFOLLOW refuses to follow a link and refuses to reuse an
    # existing inode; mode 0600 at creation closes the window where evidence
    # sat on disk world-readable between write and chmod.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    os.replace(tmp, path)


def unavailable(reason: str, seam: str) -> dict[str, str]:
    """A field that cannot be captured today, with the reason and the seam.

    ``seam`` names the exact place in the codebase where the information is
    lost, so a later slice knows where to reach rather than re-deriving it.
    """
    return {"status": "unavailable", "reason": reason, "seam": seam}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        return bool(out.strip())
    except Exception:
        return None


def new_run_id() -> str:
    """Timestamp plus a short random suffix.

    The suffix matters: two runs started in the same second would otherwise
    collide, and the loser would be silently overwritten.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:RUN_ID_SUFFIX_LEN]}"


class RunEvidence:
    """Writer for one run's evidence directory."""

    def __init__(self, output_root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.run_dir = output_root / run_id
        if self.run_dir.exists():
            # Refuse rather than merge: a half-overwritten run directory is
            # worse than no directory, because it still looks readable.
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")
        self.scenario_dir = self.run_dir / "scenarios"
        self.scenario_dir.mkdir(parents=True)
        # Evidence files carry complete specialist prompts and responses,
        # which embed the company profile and retrieved company_docs chunks.
        # Not world-readable on a shared host or container.
        #
        # Each chmod is guarded SEPARATELY: sharing one suppress meant a
        # failure on the first silently skipped the second, leaving the whole
        # tree at 0755. Filesystems that reject chmod (some bind mounts, SMB,
        # NFS) hit this without an attacker, so a failure warns rather than
        # passing for success.
        for directory in (self.run_dir, self.scenario_dir):
            try:
                os.chmod(directory, 0o700)
            except OSError as exc:
                print(
                    f"WARNING: could not restrict permissions on {directory} "
                    f"({exc}). Evidence may be readable by other local users.",
                    file=sys.stderr,
                )
        self._manifest: dict[str, Any] = {}

    def _scenario_path(self, scenario_id: str) -> Path:
        """Path for one scenario's evidence, proven to stay inside the run dir.

        Belt and braces: ``safe_filename`` already strips traversal, and this
        re-checks containment after resolution so any future change to the
        sanitizer cannot silently reopen an arbitrary-write hole.
        """
        path = (self.scenario_dir / f"{safe_filename(scenario_id)}.json").resolve()
        root = self.scenario_dir.resolve()
        if root != path.parent:
            raise ValueError(
                f"refusing to write scenario evidence outside {root}: {path}"
            )
        if path.exists():
            # Same rule as the run directory one level up: refuse rather than
            # merge. Overwriting would leave the manifest counting two
            # scenarios with only one on disk.
            raise ValueError(
                f"refusing to overwrite existing scenario evidence: {path} "
                f"(scenario id {scenario_id!r})"
            )
        return path

    # -- manifest ---------------------------------------------------------

    def open_manifest(
        self,
        *,
        kind: str,
        inventory: dict[str, Any],
        knowledge: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Write the pre-execution manifest. Status starts INCOMPLETE."""
        self._manifest = {
            "run_id": self.run_id,
            "status": STATUS_INCOMPLETE,
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
            "kind": kind,
            "git": {"sha": _git_sha(), "dirty": _git_dirty()},
            "config": config,
            "inventory": inventory,
            "knowledge": knowledge,
            "counts": {
                "discovered": inventory.get("discovered_count", 0),
                "executed": 0,
                "passed": 0,
                "failed": 0,
                "errored": 0,
            },
            "results": [],
            "infrastructure_errors": [],
        }
        self._flush()

    def _flush(self) -> None:
        # Same writer as the scenario files: atomic, symlink-safe, 0600. The
        # manifest carries argv, store paths and infrastructure_errors detail
        # (which for a provider failure includes the response body), so it
        # needs the same protection they get.
        _atomic_write(self.run_dir / "manifest.json", json.dumps(self._manifest, indent=2, default=str))

    # -- results ----------------------------------------------------------

    def record(self, result: dict[str, Any]) -> None:
        """Persist one scenario's full evidence and update the running counts."""
        scenario_id = str(result.get("id", "unknown"))
        path = self._scenario_path(scenario_id)
        _atomic_write(path, json.dumps(result, indent=2, default=str))

        outcome = result.get("outcome")
        counts = self._manifest["counts"]
        counts["executed"] += 1
        if outcome == OUTCOME_PASS:
            counts["passed"] += 1
        elif outcome == OUTCOME_FAIL:
            counts["failed"] += 1
        else:
            counts["errored"] += 1

        # The manifest keeps a compact index; the per-scenario file holds
        # everything, so the manifest stays readable at 40+ scenarios.
        self._manifest["results"].append(
            {
                "id": scenario_id,
                "domain": result.get("domain"),
                "outcome": outcome,
                "overall": (result.get("judge") or {}).get("overall"),
                "duration_ms": result.get("duration_ms"),
                "error": result.get("error"),
            }
        )
        self._flush()

    def add_infrastructure_error(self, scenario_id: str, kind: str, detail: str) -> None:
        """Record a harness/judge/model failure — never a score."""
        self._manifest["infrastructure_errors"].append(
            {"scenario_id": scenario_id, "kind": kind, "detail": detail}
        )
        self._flush()

    # -- close ------------------------------------------------------------

    def close(self) -> str:
        """Finalize status and return it.

        COMPLETE requires all three: every discovered scenario executed, no
        scenario in ERROR, and no recorded infrastructure error. Anything else
        is FAILED — a run that graded 39 of 42 scenarios is not a baseline.
        """
        counts = self._manifest["counts"]
        if counts["errored"] or self._manifest["infrastructure_errors"]:
            # Something broke in the harness/judge/model plumbing. Distinct
            # from INCOMPLETE: this run produced wrong-or-missing verdicts, it
            # did not merely stop early.
            status = STATUS_FAILED
        elif counts["executed"] < counts["discovered"]:
            # Killed, interrupted, or preflight-only. Visibly unfinished.
            status = STATUS_INCOMPLETE
        else:
            status = STATUS_COMPLETE
        self._manifest["status"] = status
        self._manifest["finished_at"] = datetime.now(UTC).isoformat()
        if counts["executed"]:
            self._manifest["counts"]["pass_rate"] = round(
                counts["passed"] / counts["executed"], 4
            )
        self._flush()
        return self._manifest["status"]

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest


# ---------------------------------------------------------------------------
# Audit harvesting
# ---------------------------------------------------------------------------

# What the audit trail genuinely cannot tell us, and where it is lost. Recorded
# verbatim into every chat result so the gap is visible in the evidence rather
# than inferred from its absence.
RETRIEVAL_PROVENANCE_GAP = {
    "chunk_ids": unavailable(
        "Chroma record ids are dropped before the audit boundary, so a "
        "retrieved passage cannot be traced back to a specific stored record.",
        "knowledge/store.py:ChromaDBStore.query builds {text, metadata, "
        "distance} and discards the ids Chroma returned; "
        "knowledge/retriever.py:_emit_retrieval_audit._chunks then projects "
        "only source/domain/distance/text_preview.",
    ),
    "full_chunk_text": unavailable(
        "Audit rows store a 400-character preview per chunk, not the passage "
        "the specialist actually read.",
        "knowledge/retriever.py:_emit_retrieval_audit._chunks truncates with "
        "(r.get('text') or '')[:400].",
    ),
    "specialist_token_usage": unavailable(
        "Per-call token counts are emitted only for Executive iterations; "
        "specialist completions report no usage anywhere.",
        "orchestrator/executive.py:_emit_cache_event is called only from "
        "_stream_agent_loop; agents/base.py:analyze discards the provider "
        "Message's usage block and returns text only.",
    ),
}


def harvest_audit(session_id: str) -> dict[str, Any]:
    """Read back what the product already recorded for one eval session.

    Purely observational: it queries rows the Executive wrote on its own during
    the turn. Nothing here changes orchestration, prompts, routing or ranking
    to manufacture provenance.

    Note the boundary honestly — these are the product's *audit* records, not
    retrieval provenance. They show which specialist was asked what and what it
    replied; they cannot show which stored records grounded that reply (see
    ``RETRIEVAL_PROVENANCE_GAP``).
    """
    out: dict[str, Any] = {
        "session_id": session_id,
        "specialists": [],
        "retrievals": [],
        "executive_usage": [],
        "gaps": RETRIEVAL_PROVENANCE_GAP,
    }
    try:
        from openexecutive.audit.logger import get_audit_logger
    except Exception as exc:
        out["error"] = f"audit logger unavailable: {exc}"
        return out

    try:
        logger = get_audit_logger()
        for event_type, bucket in (
            ("specialist_consult", "specialists"),
            ("knowledge_retrieval", "retrievals"),
            ("cache_event", "executive_usage"),
        ):
            events = logger.query(
                event_type=event_type, session_id=session_id, limit=1000
            )
            for ev in events:
                # query() intentionally leaves `full` unpopulated to keep the
                # list view small; get(id) is the only way to the untruncated
                # payload (the specialist's complete response).
                detailed = logger.get(ev.id) or ev
                out[bucket].append(
                    {
                        "id": ev.id,
                        "ts": ev.ts,
                        "actor": ev.actor,
                        "turn_id": ev.turn_id,
                        "details": detailed.details,
                        "full": detailed.full,
                    }
                )
    except Exception as exc:  # pragma: no cover - defensive
        out["error"] = f"audit harvest failed: {exc}"

    for bucket in ("specialists", "retrievals", "executive_usage"):
        out[bucket].reverse()  # query() returns newest-first; restore turn order
    return out
