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

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
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
        {"name": scorer.name, "description": scorer.description}
        for scorer in get_scorers().values()
    ]


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    min_phrases: int = Form(2),
    max_variants: int = Form(99),
    root_pos: str = Form("paninian"),
    grammar_filter: bool = Form(True),
    scorers: str = Form(""),
) -> dict:
    """Upload a corpus and start a pipeline job. Returns the job id."""
    from scoring import get_scorers

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Only {sorted(ALLOWED_SUFFIXES)} files are accepted.")
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

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File exceeds the 50 MB upload limit.")
    if not content.strip():
        raise HTTPException(400, "Uploaded file is empty.")

    job = jobs.create_job()
    input_path = jobs.job_dir(job.job_id) / f"input{suffix}"
    input_path.write_bytes(content)

    options = {
        "allowed_root_pos": ROOT_POS_PRESETS[root_pos],
        "scheme": root_pos,  # annotation scheme name — drives scheme-aware scorers
        "min_phrases": min_phrases,
        "max_variants": max_variants,
        "grammar_filter": grammar_filter,
        "scorers": scorer_names,
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


# Static frontend — mounted last so /api/* keeps priority.
from webapp.taru_routes import router as taru_router
app.include_router(taru_router)
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True),
    name="static",
)
