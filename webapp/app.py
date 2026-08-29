"""
webapp.app
==========
FastAPI routes for the Hindi NLP Toolkit web interface.  Routes only —
all NLP work lives in the logic packages and webapp.pipeline.

Run from the project root:

    python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000

Interactive API docs: http://localhost:8000/docs
"""

from __future__ import annotations

import io
import os
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from webapp import jobs, ratelimit
from webapp.pipeline import run_job
from webapp.warmup import start_background_warmup

ALLOWED_SUFFIXES = {".txt", ".conllu"}
# 250 MB: five times the largest treebank we ship (hi_hdtb-ud-train.conllu is
# 46.5 MB), so any realistic corpus fits, while still bounding how long a single
# upload can occupy one of the two workers.  Overridable without a rebuild.
MAX_UPLOAD_BYTES = int(os.environ.get("HNT_MAX_UPLOAD_BYTES", 250 * 1024 * 1024))
_UPLOAD_CHUNK_SIZE = 1024 * 1024
# Sentence mode is one sentence plus one context sentence; anything beyond this
# is a mistake or an abuse.  Starlette caps non-file form fields at 1 MB anyway,
# but that surfaces as FastAPI's generic "error parsing the body" 400 — this cap
# is reached first and says something useful.
MAX_TEXT_CHARS = 2000

ROOT_POS_PRESETS = {
    "paninian": None,  # library defaults: VERB / AUX / VM / VAUX
    # Verb-only, per Ranjan & van Schijndel (2024) §Data and Methods, criterion
    # (d): "the root node for each tree is a finite verb with at least two
    # preverbal dependents".  UD makes the *predicate* the root of a copular
    # clause, so NOUN/ADJ/PROPN roots were once allowed here — but permuting the
    # dependents of a nominal root is not the paper's construct (there is no
    # root verb to be "preverbal" to).  On UD-HDTB this keeps the 87.6% of trees
    # that are VERB-rooted and drops 1,546 / 13,306 copular ones.
    # The *library* stays flexible: filter_bad_root(allowed_pos=...) still
    # accepts any tag set, so callers who do want copular clauses can pass them.
    "ud": ["VERB", "AUX"],
}

MEDIA_TYPES = {
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".conllu": "text/plain; charset=utf-8",
}

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Pre-warm Stanza + LSTM + trigram on a background thread so the first
    # user job is fast (opt out with HNT_NO_PREWARM=1) — see webapp/warmup.py.
    start_background_warmup()
    yield


app = FastAPI(
    title="Hindi NLP Toolkit",
    description="Filter Hindi sentences and generate preverbal constituent "
    "order variants. Upload .conllu or .txt, download results per stage.",
    lifespan=_lifespan,
)


class MaxBodySizeMiddleware:
    """Reject an oversized upload from Content-Length, before anything reads it.

    This is the one piece of middleware in the app, and it exists because a
    check inside ``create_job`` cannot be early enough.  FastAPI resolves
    File/Form parameters by calling ``await request.form()`` (fastapi/routing.py)
    *before* it solves dependencies or enters the handler, and Starlette's
    multipart parser applies no size limit at all to file parts — it spools them
    straight to a temporary file.  So by the time any route code runs, the whole
    body has already been received.  Only an ASGI middleware, which runs before
    routing, can refuse it up front.

    Scoped to exactly one method+path, so it can never touch another endpoint —
    in particular never the 1.5 s status poller.  Content-Length is
    client-supplied and may be absent or wrong; this is a cheap fast path, and
    ``_stream_upload`` remains the authoritative check.
    """

    # Multipart framing (boundaries, the other form fields) rides along with the
    # file, so allow some slack before calling a request oversized.
    SLACK_BYTES = 64 * 1024

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/jobs"
        ):
            for name, value in scope.get("headers", []):
                if name == b"content-length":
                    try:
                        too_big = int(value) > MAX_UPLOAD_BYTES + self.SLACK_BYTES
                    except ValueError:
                        too_big = False
                    if too_big:
                        response = JSONResponse({"detail": _too_large_message()}, status_code=413)
                        await response(scope, receive, send)
                        return
                    break
        await self.app(scope, receive, send)


app.add_middleware(MaxBodySizeMiddleware)
# Added last, so it wraps the body guard too and every reply carries the cookie
# — including the 413 above and the limiter's own 429.
app.add_middleware(ratelimit.SessionMiddleware)


def _too_large_message() -> str:
    return f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."


async def _stream_upload(file: UploadFile, dest: Path) -> tuple[bool, bool]:
    """Copy an upload to disk in bounded chunks, refusing to exceed the cap.

    Returns ``(within_cap, has_content)``.  On a cap breach the partial file is
    removed and ``(False, False)`` comes back.  Counting as we go is what makes
    the limit real: Content-Length can be absent or understated, so the
    middleware above is only a fast path, and this loop is what actually bounds
    what we write and what we hold in memory.

    ``has_content`` tracks non-whitespace as it streams, preserving the old
    ``content.strip()`` check without ever holding the whole file.
    """
    total = 0
    has_content = False
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    break
                has_content = has_content or bool(chunk.strip())
                out.write(chunk)
        if total > MAX_UPLOAD_BYTES:
            dest.unlink(missing_ok=True)
            return False, False
        return True, has_content
    except Exception:
        dest.unlink(missing_ok=True)
        raise


@app.get("/api/plugins")
def list_plugins() -> list:
    """Scorer plugins discovered in the scoring/ package."""
    from scoring import get_scorers

    return [
        {
            "name": scorer.name,
            # Fallback baked in here so the UI never has to handle an empty one.
            "display_name": getattr(scorer, "display_name", "") or scorer.name,
            "description": scorer.description,
            "trained_on": getattr(scorer, "trained_on", ""),
            "built_with": getattr(scorer, "built_with", ""),
            "notes": getattr(scorer, "notes", ""),
            "needs_previous_sentence": getattr(scorer, "needs_previous_sentence", False),
        }
        for scorer in get_scorers().values()
    ]


@app.post("/api/jobs")
async def create_job(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    context_text: str = Form(""),
    min_phrases: int = Form(2),
    max_variants: int = Form(99),
    root_pos: str = Form("paninian"),
    scorers: str = Form(""),
    evaluate: bool = Form(False),
    recommend: bool = Form(True),
    _keys: tuple = Depends(
        ratelimit.guard(
            "jobs",
            session_limit=ratelimit.JOBS_SESSION_PER_MIN,
            ip_limit=ratelimit.JOBS_IP_PER_MIN,
        )
    ),
) -> dict:
    """
    Start a pipeline job from either an uploaded corpus *or* typed sentences.

    Exactly one input is expected:
      - ``file``: a ``.conllu`` / ``.txt`` upload (uses ``root_pos`` as posted).
      - ``text``: a single Hindi sentence, parsed by Stanza (UD). Typed input
        forces the UD scheme and runs with the grammar filter off so the one
        sentence still yields variants. An optional ``context_text`` (the
        preceding sentence) feeds the context-aware scorers; without it, scorers
        that need a preceding sentence (``needs_previous_sentence``) are skipped.

    ``evaluate`` (File mode only) appends a pairwise ranking-accuracy stage
    over the scorers' delta columns; it is forced off for typed text, where a
    single sentence cannot yield enough pairs.

    ``recommend`` (Sentence mode only, on by default) gets a word-order
    recommendation instead: the reference and its variants are ranked by the
    ticked scorers' features and the most natural order is written to
    recommendation.json (see webapp/pipeline.py).  Unticking it also drops the
    Recommended_* columns from variants.csv.  File mode has no such control and
    keeps those columns.

    Returns the job id.
    """
    from scoring import get_scorers

    has_file = file is not None and bool(file.filename)
    has_text = bool(text.strip())
    if has_file and has_text:
        raise HTTPException(400, "Provide either a file or typed text, not both.")
    if not has_file and not has_text:
        raise HTTPException(400, "Provide a file or type at least one sentence.")
    if root_pos not in ROOT_POS_PRESETS:
        raise HTTPException(400, f"root_pos must be one of {sorted(ROOT_POS_PRESETS)}.")
    if min_phrases < 1:
        raise HTTPException(400, "min_phrases must be >= 1.")
    if max_variants < 1:
        raise HTTPException(400, "max_variants must be >= 1.")

    scorer_names = [s.strip() for s in scorers.split(",") if s.strip()]
    unknown = set(scorer_names) - set(get_scorers())
    if unknown:
        raise HTTPException(400, f"Unknown scorer(s): {sorted(unknown)}.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"Sentence is too long (limit {MAX_TEXT_CHARS} characters).")
    if len(context_text) > MAX_TEXT_CHARS:
        raise HTTPException(
            400, f"Context sentence is too long (limit {MAX_TEXT_CHARS} characters)."
        )
    suffix = Path(file.filename or "").suffix.lower() if has_file else ""
    if has_file and suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Only {sorted(ALLOWED_SUFFIXES)} files are accepted.")

    # Every cheap check is now behind us, so the job below is only registered
    # once the request is known to be worth a worker slot.
    session_key, ip_key = _keys
    job = jobs.try_create_job(
        session_key,
        ip_key,
        session_max=ratelimit.JOBS_SESSION_INFLIGHT if ratelimit.enabled() else 0,
        ip_max=ratelimit.JOBS_IP_INFLIGHT if ratelimit.enabled() else 0,
    )
    if job is None:
        raise HTTPException(429, ratelimit.BUSY_MESSAGE, headers={"Retry-After": "10"})
    context_sentence = context_text.strip()

    if has_text:
        # A single typed sentence → Stanza (UD). Route through run_job's .txt
        # branch and drop the corpus-bigram grammar filter so the one sentence
        # still permutes. Collapse stray whitespace/line breaks into one line.
        scheme = "ud"
        grammar_filter = False
        evaluate = False  # one sentence can never reach the pair threshold
        target = " ".join(text.split())
        input_path = jobs.job_dir(job.job_id) / "input.txt"
        input_path.write_text(target, encoding="utf-8")
        # No context sentence → skip scorers that need a preceding sentence.
        if not context_sentence:
            registry = get_scorers()
            scorer_names = [
                n for n in scorer_names
                if not getattr(registry[n], "needs_previous_sentence", False)
            ]
    else:
        scheme = root_pos
        grammar_filter = True
        input_path = jobs.job_dir(job.job_id) / f"input{suffix}"
        within_cap, has_content = await _stream_upload(file, input_path)
        if not within_cap:
            jobs.discard(job)
            raise HTTPException(413, _too_large_message())
        if not has_content:
            jobs.discard(job)
            raise HTTPException(400, "Uploaded file is empty.")

    options = {
        "allowed_root_pos": ROOT_POS_PRESETS[scheme],
        "scheme": scheme,  # annotation scheme name — drives scheme-aware scorers
        "min_phrases": min_phrases,
        "max_variants": max_variants,
        "grammar_filter": grammar_filter,
        "scorers": scorer_names,
        "evaluate": evaluate,
        # Ranking work behind the Recommended_* columns of variants.csv.  The
        # opt-out is Sentence mode's; File mode has always carried the columns
        # and keeps them.
        "recommend": recommend if has_text else True,
        # Sentence-mode headline result (recommendation.json + its card); File
        # mode has `evaluate` instead.
        "recommend_json": has_text and recommend,
        "context_text": context_sentence if has_text else "",
    }
    jobs.submit(job, run_job, job, input_path, options)
    return {"job_id": job.job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (it may have expired).")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Stop a queued or running job and return its final state.

    Returns as soon as the killer has fired: the worker unwinds on its next
    heartbeat, in the background, so the client is free to submit a new job
    immediately.  Artifacts already written stay downloadable.  Idempotent —
    cancelling a finished job is a no-op that reports the existing state.

    Like the download routes, the 12-hex job id is the only credential; a
    client can therefore only cancel a job whose id it was given.
    """
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (it may have expired).")
    if job.status not in ("done", "failed", "cancelled"):
        job.cancel()
    return job.to_dict()


@app.get("/api/jobs/{job_id}/files/{filename}")
def download_artifact(job_id: str, filename: str) -> FileResponse:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (it may have expired).")
    # Only names the pipeline registered are servable — no path traversal.
    if filename not in job.artifacts:
        raise HTTPException(404, f"Artifact '{filename}' is not (yet) available.")
    path = jobs.job_dir(job_id) / filename
    if not path.exists():
        raise HTTPException(404, "Artifact file missing on disk.")
    return FileResponse(
        path,
        filename=filename,
        media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
    )


@app.get("/api/jobs/{job_id}/files.zip")
def download_all(job_id: str) -> Response:
    """Bundle every available artifact for a job into a single .zip."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (it may have expired).")
    jdir = jobs.job_dir(job_id)
    present = [name for name in job.artifacts if (jdir / name).exists()]
    if not present:
        raise HTTPException(404, "No artifacts available yet.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in present:
            zf.write(jdir / name, arcname=name)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="hindi-nlp-{job_id}.zip"'},
    )


# Hand-written docs page — intentionally not linked from the SPA.
@app.get("/how-it-works", include_in_schema=False)
def how_it_works() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "how-it-works.html")


# Static frontend — mounted last so /api/* keeps priority.
from webapp.taru_routes import router as taru_router
app.include_router(taru_router)
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True),
    name="static",
)
