"""
webapp/taru_routes.py
=====================
Mounts the "Taru" Hindi Tree & Surprisal tool into the HindiNLPToolkit
FastAPI app as a self-contained sub-app under /taru — no second server, no
Hugging Face redirect.

It reuses the existing pipeline (Berkeley parser + synproc + training) that
lives in the `taru/` package (a copy of the SyntacticTreeSurprisal workspace:
flask_server.py's logic, train_pipeline.py, dep2linetrees.py, _parse_one.py,
bin/, genmodel/, scripts/, and the sibling resource-* dirs).

Wiring (one line in webapp/app.py):

    from webapp.taru_routes import router as taru_router
    app.include_router(taru_router)

Then the toolkit card's button links to /taru.
"""
from __future__ import annotations

import sys
import uuid
import threading
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, Body, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

# --- locate the bundled Taru workspace and make it importable ----------------
# Expected layout inside the toolkit repo:
#   taru/workspace/   <- flask logic, train_pipeline.py, dep2linetrees.py, _parse_one.py, bin/, genmodel/, scripts/
#   taru/resource-lcparse/  taru/resource-linetrees/  taru/resource-incrsem/ ...
#   taru/external_resources/berkeleyparser/berkeleyParser.jar
TARU_ROOT = Path(__file__).resolve().parent.parent / "taru"
TARU_WS = TARU_ROOT / "workspace"
if str(TARU_WS) not in sys.path:
    sys.path.insert(0, str(TARU_WS))      # so `import train_pipeline`, `import dep2linetrees` resolve

# Import the pipeline pieces. We pull the parse/train functions out of the
# Taru backend. To avoid running Flask, the parse/train *logic* lives in
# taru_backend.py (a Flask-free copy of the functions from flask_server.py).
import taru_backend as tb   # noqa: E402  (path set above)

router = APIRouter(prefix="/taru", tags=["taru"])

JOBS: Dict[str, dict] = {}


# ---------------- page ----------------

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def taru_page() -> HTMLResponse:
    html = (TARU_WS / "taru_viewer.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# ---------------- models ----------------

@router.get("/models")
def models() -> dict:
    return {mid: {"label": m["label"]} for mid, m in tb.get_models().items()}


# ---------------- parse ----------------

@router.post("/parse")
def parse(payload: dict = Body(...)) -> object:
    raw = (payload.get("sentence") or "").strip()
    model_id = payload.get("model", "hdtb")
    want_surp = bool(payload.get("surprisal", True))
    input_type = payload.get("input_type", "sentence")
    if not raw:
        raise HTTPException(400, "empty input")
    try:
        if input_type == "dependency":
            results = tb.parse_dependency(raw, model_id, want_surp)
        else:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            results = [tb.parse_one(ln, model_id, want_surp) for ln in lines]
        return results if len(results) > 1 else results[0]
    except Exception as e:  # surface the real error to the UI
        raise HTTPException(500, str(e))


# ---------------- train ----------------

def _train_job(job_id: str, text: str, name: str, input_type: str) -> None:
    job = JOBS[job_id]
    log = lambda m: job["log"].append(m)
    try:
        if input_type == "dependency":
            info = tb.train_from_conllu(text, name, log=log)
        else:
            info = tb.train_from_raw(text, name, log=log)
        job.update(status="done", model=info)
    except Exception as e:
        job.update(status="error", error=str(e))


@router.post("/train")
async def train(request: Request) -> dict:
    """Accept either a JSON body {text,name,input_type} (what taru_viewer.html
    sends) or a multipart form with an uploaded file."""
    text, name, input_type = "", "", "sentence"
    ctype = request.headers.get("content-type", "")

    if ctype.startswith("application/json"):
        body = await request.json()
        text = body.get("text", "")
        name = body.get("name", "")
        input_type = body.get("input_type", "sentence")
    else:
        form = await request.form()
        name = form.get("name", "")
        input_type = form.get("input_type", "sentence")
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            text = (await upload.read()).decode("utf-8", errors="replace")
        else:
            text = form.get("text", "")

    if not text.strip():
        raise HTTPException(400, "no training text provided")
    name = tb.safe_name(name or f"model_{uuid.uuid4().hex[:6]}")
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "log": [], "name": name}
    threading.Thread(target=_train_job, args=(job_id, text, name, input_type),
                     daemon=True).start()
    return {"job_id": job_id, "name": name}


@router.get("/train/status/{job_id}")
def train_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


@router.get("/download/{model_id}")
def download(model_id: str) -> FileResponse:
    m = tb.list_custom_models().get(model_id)
    if not m or not Path(m["zip"]).exists():
        raise HTTPException(404, "no downloadable package for this model")
    return FileResponse(m["zip"], filename=f"{model_id}_model_package.zip",
                        media_type="application/zip")