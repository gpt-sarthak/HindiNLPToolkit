#!/usr/bin/env python3
"""
train_pipeline.py — train a custom constituency + surprisal model from RAW text.

Flow (mirrors the documented HDTB training journey, generalized):
  1. raw sentences (.txt, one per line)
  2. -> parse with the BASE model (HDTB) to generate trees  => {name}.linetrees  (silver training data)
  3. -> Berkeley GrammarTrainer on those trees              => model.gr           (parsing grammar)
  4. -> WriteGrammarToTextFile                              => model.grammar/.lexicon
  5. -> berkgrammar2ckygr_auto (root auto-detected S_0 fix) +
        berklexicon2ckylex -> ccu2cc -> ccmodel2synproccptmodel
                                                            => model.synprocmodel (surprisal model)
Outputs land in genmodel/custom/{name}/ and are zipped for download.

NOTE: trees generated from raw text are SILVER (produced by the base parser),
not gold annotation. The trained model learns the base parser's analyses of
the user's domain — useful for domain adaptation, not a substitute for a
hand-annotated treebank.
"""
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # workspace/
ROOT = HERE.parent                               # model_blocks_english root
JAR = ROOT / "external_resources/berkeleyparser/berkeleyParser.jar"
LCPARSE = ROOT / "resource-lcparse/scripts"
AUTO_CONV = HERE / "scripts/berkgrammar2ckygr_auto.py"
CCMODEL_BIN = HERE / "bin/ccmodel2synproccptmodel"
BASE_GR = HERE / "genmodel/training/hindi/hdtb_fresh.gr"
CUSTOM_DIR = HERE / "genmodel/custom"

JAVA_MEM = os.environ.get("TRAIN_JAVA_MEM", "2g")   # modest default for HF free tier
SM_CYCLES = int(os.environ.get("TRAIN_SM_CYCLES", "1"))  # 1 = fast; raise for quality


def _run(cmd, step, log, **kw):
    log(step)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{step} failed:\n{(r.stderr or r.stdout)[-2000:]}")
    return r


def _sh(cmd, step, log):
    log(step)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{step} failed:\n{(r.stderr or r.stdout)[-2000:]}")
    return r


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())[:40]
    return name or "custom_model"


def _train_from_trees(trees, name, raw_source, source_kind, log):
    """Shared core: given linetrees, train grammar + synprocmodel + package."""
    name = safe_name(name)
    out = CUSTOM_DIR / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    good = [t for t in trees if t.strip().startswith("(") and "(())" not in t]
    if len(good) < 5:
        raise RuntimeError("Too few valid trees to train (need >= 5).")
    trees_path = out / f"{name}.linetrees"
    trees_path.write_text("\n".join(good) + "\n", encoding="utf-8")
    (out / f"{name}.source.txt").write_text(raw_source, encoding="utf-8")

    # train a Berkeley grammar on the trees
    gr_path = out / "model.gr"
    _run(["java", f"-Xmx{JAVA_MEM}", "-cp", str(JAR),
          "edu.berkeley.nlp.PCFGLA.GrammarTrainer",
          "-SMcycles", str(SM_CYCLES), "-smooth", "SmoothAcrossParentBits",
          "-path", str(trees_path), "-treebank", "SINGLEFILE", "-out", str(gr_path)],
         f"Training Berkeley grammar ({SM_CYCLES} SM cycle(s)) on {len(good)} trees", log)

    _run(["java", f"-Xmx{JAVA_MEM}", "-cp", str(JAR),
          "edu.berkeley.nlp.PCFGLA.WriteGrammarToTextFile",
          str(gr_path), str(out / "model")],
         "Writing grammar/lexicon text files", log)

    ccu = out / "model.x-ccu.model"
    _sh(f"cat '{out}/model.grammar' | sed 's/[+]/-/g' | python3 '{AUTO_CONV}' > '{ccu}'",
        "Converting grammar (auto root detection)", log)
    _sh(f"cat '{out}/model.lexicon' | sed 's/[+]/-/g' | "
        f"python3 '{LCPARSE}/berklexicon2ckylex.py' >> '{ccu}'", "Converting lexicon", log)
    cc = out / "model.x-cc.model"
    _sh(f"cat '{ccu}' | python3 '{LCPARSE}/ccu2cc.py' > '{cc}'", "Binarizing rules", log)
    synmodel = out / "model.synprocmodel"
    _sh(f"cat '{cc}' | sed 's/^Cr/R/' | '{CCMODEL_BIN}' > '{synmodel}'",
        "Building synproc surprisal model", log)
    if synmodel.stat().st_size < 1000:
        raise RuntimeError("synprocmodel came out empty - conversion failed.")

    zip_path = out / f"{name}_model_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [gr_path, synmodel, trees_path, out / f"{name}.source.txt",
                  out / "model.grammar", out / "model.lexicon"]:
            if f.exists():
                z.write(f, arcname=f.name)
    log("Done")
    return {"id": name, "dir": str(out), "grammar": str(gr_path),
            "synproc": str(synmodel), "linetrees": str(trees_path),
            "zip": str(zip_path), "n_sentences": len(good), "source": source_kind}


def train_from_raw(raw_text, name, log=lambda s: None):
    """Train from RAW sentences: parse with base model -> silver trees -> train."""
    sentences = [s.strip() for s in raw_text.splitlines() if s.strip()]
    if len(sentences) < 5:
        raise ValueError("Need at least 5 sentences to train a model.")
    tmp = CUSTOM_DIR / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    raw_path = tmp / "raw.txt"
    raw_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    trees_path = tmp / "raw.linetrees"
    _run(["java", f"-Xmx{JAVA_MEM}", "-cp", str(JAR),
          "edu.berkeley.nlp.PCFGLA.BerkeleyParser", "-gr", str(BASE_GR),
          "-inputFile", str(raw_path), "-outputFile", str(trees_path)],
         f"Parsing {len(sentences)} sentences into silver trees (base model)", log)
    trees = trees_path.read_text(encoding="utf-8").splitlines()
    return _train_from_trees(trees, name, raw_text, "raw_sentences (silver trees)", log)


def train_from_conllu(conllu_text, name, log=lambda s: None):
    """Train from DEPENDENCY (CoNLL-U) input: convert to gold trees -> train."""
    import dep2linetrees as d2l
    log("Converting dependency trees to constituency linetrees")
    trees = d2l.conllu_to_trees(conllu_text)
    if len(trees) < 5:
        raise ValueError("Got fewer than 5 trees from the dependency input.")
    return _train_from_trees(trees, name, conllu_text, "dependency (gold trees)", log)


def list_custom_models() -> dict:
    models = {}
    if CUSTOM_DIR.exists():
        for d in sorted(CUSTOM_DIR.iterdir()):
            gr, sp = d / "model.gr", d / "model.synprocmodel"
            if gr.exists() and sp.exists():
                models[d.name] = {"grammar": str(gr), "synproc": str(sp),
                                  "zip": str(d / f"{d.name}_model_package.zip")}
    return models
