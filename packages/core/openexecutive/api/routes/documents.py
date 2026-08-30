from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile

from openexecutive.api.models import CompanyDocContent, DocumentUploadResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".txt"}


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile,
    # `Form(...)` (not a bare default) so FastAPI reads `domain` from the
    # multipart body the UI sends. A bare `domain: str = "general"` is parsed
    # as a query param, so the form field is dropped and every upload lands
    # under "general" — invisible to domain-filtered specialist retrieval.
    domain: str = Form("general"),
    request: Request = None,  # type: ignore[assignment]
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Strip directory components to prevent path traversal (e.g. "../../etc/passwd.md")
    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Reject names the filesystem cannot represent BEFORE anything is indexed.
    # Vectors are written before the persistent file, so a name that passes the
    # extension allowlist but blows up `dest.write_bytes` (an embedded NUL, or a
    # component over the OS limit) would commit chunks for a document that then
    # has no file — invisible to GET /documents and permanently undeletable,
    # since DELETE 404s on the missing file. That is a durable injection
    # primitive, so it is refused up front rather than half-committed.
    if "\x00" in safe_filename or len(safe_filename.encode("utf-8")) > 255:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from openexecutive.config import get_settings
        from openexecutive.knowledge.loader import company_document_id, ingest_file
        from openexecutive.knowledge.store import ChromaDBStore

        settings = get_settings()
        store = (
            request.app.state.store
            if request and hasattr(request.app.state, "store")
            else ChromaDBStore(persist_directory=settings.vector_store_path)
        )

        # `tmp_path` is only where the bytes live for extraction. Identity comes
        # from `safe_filename` — the same key the destination path below, the
        # listing, and DELETE all use — so a re-upload replaces this document
        # instead of accumulating a second copy under a fresh temp name.
        chunks_indexed = await ingest_file(
            path=tmp_path,
            store=store,
            domain=domain,
            collection=ChromaDBStore.COMPANY_COLLECTION,
            display_filename=safe_filename,
            document_id=company_document_id(safe_filename),
        )

        company_docs_dir = settings.company_profile_path.parent / "docs"
        company_docs_dir.mkdir(parents=True, exist_ok=True)
        dest = company_docs_dir / safe_filename
        dest.write_bytes(content)

        # Fire the proactive-alerts pipeline. Body is a best-effort excerpt
        # for triage context; PDFs/docx won't decode cleanly and that's fine —
        # the triage prompt still sees source, title, and domain.
        try:
            from openexecutive.alerts.models import AlertEvent
            from openexecutive.alerts.pipeline import schedule_evaluation

            excerpt = ""
            if ext in {".md", ".txt"}:
                excerpt = content[:8000].decode("utf-8", errors="replace")
            else:
                excerpt = f"Newly ingested {ext} document: {safe_filename} (domain: {domain})"

            schedule_evaluation(
                AlertEvent(
                    source="document",
                    external_id=safe_filename,
                    title=safe_filename,
                    body=excerpt,
                )
            )
        except Exception:
            # Never let an alert failure 500 the upload.
            import logging

            logging.getLogger(__name__).exception(
                "Failed to schedule alert evaluation for document upload"
            )

    finally:
        tmp_path.unlink(missing_ok=True)

    return DocumentUploadResponse(
        filename=safe_filename,
        chunks_indexed=chunks_indexed,
        domain=domain,
        status="indexed",
    )


@router.get("/documents")
async def list_documents(request: Request = None) -> dict:  # type: ignore[assignment]
    from openexecutive.config import get_settings
    from openexecutive.knowledge.loader import list_company_docs

    settings = get_settings()
    docs_dir = settings.company_profile_path.parent / "docs"
    return {"documents": list_company_docs(docs_dir)}


@router.get("/documents/{filename}", response_model=CompanyDocContent)
async def get_document(
    filename: str,
    request: Request = None,  # type: ignore[assignment]
) -> CompanyDocContent:
    # Same sanitization as delete: reject anything that isn't a bare filename
    # so a crafted path can't escape the docs directory.
    safe = Path(filename).name
    if not safe or safe.startswith(".") or safe != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    from openexecutive.config import get_settings
    from openexecutive.knowledge.loader import extract_text_from_file

    settings = get_settings()
    docs_dir = settings.company_profile_path.parent / "docs"
    path = docs_dir / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    # Show the extracted text — exactly what gets chunked into the vector store
    # and retrieved by the Executive. Works uniformly across PDF/DOCX/MD/TXT.
    content = extract_text_from_file(path)
    if not content.strip():
        content = "_No extractable text in this document (it may be a scanned or image-only file)._"
    return CompanyDocContent(filename=safe, content=content)


@router.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    request: Request = None,  # type: ignore[assignment]
) -> dict:
    safe = Path(filename).name
    if not safe or safe.startswith(".") or safe != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    from openexecutive.config import get_settings
    from openexecutive.knowledge.loader import company_document_id
    from openexecutive.knowledge.store import ChromaDBStore

    settings = get_settings()
    docs_dir = settings.company_profile_path.parent / "docs"
    path = docs_dir / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    store = (
        request.app.state.store
        if request and hasattr(request.app.state, "store")
        else ChromaDBStore(persist_directory=settings.vector_store_path)
    )
    # Delete by logical document identity, NOT by the display filename. Matching
    # on `filename` would also sweep away chat-attachment chunks that happen to
    # share a name — a different document the caller never addressed. The
    # namespaced document_id cannot collide across ingest paths.
    #
    # `strict=True` is what makes the 200 honest. The default best-effort mode
    # swallows every failure, so a locked/corrupt/read-only store would return
    # `{"deleted": …}` after unlinking the file while every chunk stayed live and
    # retrievable — and unreachable forever, because the next DELETE 404s on the
    # missing file. "Delete my data" must not silently mean "hide it from the UI
    # and keep feeding it to the model". Raising here aborts before the unlink,
    # so the document stays listed and the request can be retried.
    store.delete_documents(
        collection=ChromaDBStore.COMPANY_COLLECTION,
        where={"document_id": company_document_id(safe)},
        strict=True,
    )
    # Vector rows first, then the file: a failed unlink leaves a listed document
    # that can be deleted again, whereas the reverse leaves a file with no
    # vectors and no way to notice. `missing_ok` because two concurrent deletes
    # (or a delete racing an upload's write) would otherwise 500 after the rows
    # are already gone. Neither step is transactional — see
    # `knowledge.known_defects`.
    path.unlink(missing_ok=True)
    return {"deleted": safe}
