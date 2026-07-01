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
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from webapp import jobs
from webapp.pipeline import run_job

ALLOWED_SUFFIXES = {".txt", ".conllu"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # generous enough for a full treebank

ROOT_POS_PRESETS = {
    "paninian": None,  # library defaults: VERB / AUX / VM / VAUX
    "ud": ["VERB", "AUX", "NOUN", "ADJ", "PROPN"],
}

MEDIA_TYPES = {
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".conllu": "text/plain; charset=utf-8",
}

app = FastAPI(
    title="Hindi NLP Toolkit",
    description="Filter Hindi sentences and generate preverbal constituent "
    "order variants. Upload .conllu or .txt, download results per stage.",
)


@app.get("/api/plugins")
def list_plugins() -> list:
    """Scorer plugins discovered in the scoring/ package."""
    from scoring import get_scorers

    return [
        {
            "name": scorer.name,
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

    job = jobs.create_job()
    context_sentence = context_text.strip()

    if has_text:
        # A single typed sentence → Stanza (UD). Route through run_job's .txt
        # branch and drop the corpus-bigram grammar filter so the one sentence
        # still permutes. Collapse stray whitespace/line breaks into one line.
        scheme = "ud"
        grammar_filter = False
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
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"Only {sorted(ALLOWED_SUFFIXES)} files are accepted.")
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File exceeds the 50 MB upload limit.")
        if not content.strip():
            raise HTTPException(400, "Uploaded file is empty.")
        scheme = root_pos
        grammar_filter = True
        input_path = jobs.job_dir(job.job_id) / f"input{suffix}"
        input_path.write_bytes(content)

    options = {
        "allowed_root_pos": ROOT_POS_PRESETS[scheme],
        "scheme": scheme,  # annotation scheme name — drives scheme-aware scorers
        "min_phrases": min_phrases,
        "max_variants": max_variants,
        "grammar_filter": grammar_filter,
        "scorers": scorer_names,
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


# Static frontend — mounted last so /api/* keeps priority.
from webapp.taru_routes import router as taru_router
app.include_router(taru_router)
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True),
    name="static",
)
