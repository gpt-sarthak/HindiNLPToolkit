"""
taru_backend.py — Flask-free core of the Hindi Tree & Surprisal tool.

This is flask_server.py with the Flask app stripped out: just the functions
(get_models / parse_one / parse_dependency / _surprisal_for) plus re-exports
of the training functions, so any web layer (FastAPI, Flask, CLI) can import
and call them. Lives in taru/workspace/ alongside _parse_one.py, bin/, etc.
"""
import os
import json
import uuid
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent          # taru/workspace
ROOT = HERE.parent                               # taru/
os.chdir(HERE)
Path("results").mkdir(exist_ok=True)

import train_pipeline as tp
from train_pipeline import (                     # re-export for the router
    train_from_raw, train_from_conllu, list_custom_models, safe_name,
)

JAR = ROOT / "external_resources/berkeleyparser/berkeleyParser.jar"
DELIM = ROOT / "resource-linetrees/scripts/linetoks2delimlinetoks.py"
SYNPROC = HERE / "bin/synproc"
JAVA_MEM = os.environ.get("PARSE_JAVA_MEM", "2g")

BUILTIN = {
    "hdtb": {
        "label": "HDTB (default, trained on Hindi Dependency Treebank)",
        "grammar": str(HERE / "genmodel/training/hindi/hdtb_fresh.gr"),
        "synproc": str(HERE / "genmodel/model/hindi/hdtb_FIXED.synprocmodel"),
    }
}


def get_models():
    models = dict(BUILTIN)
    for mid, info in tp.list_custom_models().items():
        models[mid] = {"label": f"{mid} (user-trained)", **info}
    return models


def _surprisal_for(sentence, model_id, want_surprisal):
    if not want_surprisal:
        return None
    m = get_models()[model_id]
    uid = uuid.uuid4().hex[:10]
    tok_file = HERE / f"results/sup_{uid}.tokdecs"
    os.system(f'echo "{sentence}" | python3 "{DELIM}" | '
              f'"{SYNPROC}" -p1 -c -b500 "{m["synproc"]}" > "{tok_file}" 2>/dev/null')
    r = subprocess.run(["python3", "_parse_one.py", sentence, str(tok_file),
                        "/dev/null"], capture_output=True, text=True, timeout=120)
    try:
        tok_file.unlink()
    except OSError:
        pass
    try:
        return json.loads(r.stdout.strip()).get("surprisal")
    except Exception:
        return None


def parse_one(sentence: str, model_id: str = "hdtb", want_surprisal: bool = True) -> dict:
    models = get_models()
    if model_id not in models:
        raise ValueError(f"unknown model '{model_id}'. Available: {list(models)}")
    m = models[model_id]
    uid = uuid.uuid4().hex[:10]
    sentence = sentence.strip()

    tree_file = HERE / f"results/sent_{uid}.tree"
    os.system(f'echo "{sentence}" | java -Xmx{JAVA_MEM} -cp "{JAR}" '
              f'edu.berkeley.nlp.PCFGLA.BerkeleyParser -gr "{m["grammar"]}" '
              f'> "{tree_file}" 2>/dev/null')

    tok_file = HERE / f"results/sent_{uid}.tokdecs"
    if want_surprisal:
        os.system(f'echo "{sentence}" | python3 "{DELIM}" | '
                  f'"{SYNPROC}" -p1 -c -b500 "{m["synproc"]}" > "{tok_file}" 2>/dev/null')
    else:
        tok_file.write_text("")

    r = subprocess.run(["python3", "_parse_one.py", sentence,
                        str(tok_file), str(tree_file)],
                       capture_output=True, text=True, timeout=120)
    for f in (tree_file, tok_file):
        try:
            f.unlink()
        except OSError:
            pass
    raw = r.stdout.strip()
    if not raw:
        raise RuntimeError("parser returned nothing (check model files / inputs)")
    obj = json.loads(raw)
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    obj.setdefault("sentence", sentence)
    obj["model"] = model_id
    obj["source"] = "sentence"
    if not want_surprisal:
        obj["surprisal"] = None
    return obj


def parse_dependency(conllu_text: str, model_id: str = "hdtb",
                     want_surprisal: bool = True) -> list:
    import dep2linetrees as d2l
    results = []
    for sent in d2l.read_conllu_text(conllu_text):
        tree = d2l.convert(sent)
        if not tree:
            continue
        sentence = " ".join(t["word"] for t in sent)
        surp = _surprisal_for(sentence, model_id, want_surprisal)
        results.append({"sentence": sentence, "tree": tree, "surprisal": surp,
                        "model": model_id, "source": "dependency"})
    if not results:
        raise RuntimeError("no valid trees parsed from the dependency input")
    return results
