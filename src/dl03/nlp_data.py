"""Datos de texto: hojas de vida de Jarvis Calling Hiring Contest (Kaggle).

Esquema esperado de ``Resume/Resume.csv``: ID, Resume_str, Resume_html, Category (24 categorías de cargo).
Se modela ``Resume_str``; ``Resume_html`` es la misma información con marcado y no se usa.
"""
from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import StratifiedGroupKFold

from .config import ExperimentConfig
from .seg_data import sha256

CATEGORIES = ["HR", "DESIGNER", "INFORMATION-TECHNOLOGY", "TEACHER", "ADVOCATE", "BUSINESS-DEVELOPMENT",
              "HEALTHCARE", "FITNESS", "AGRICULTURE", "BPO", "SALES", "CONSULTANT", "DIGITAL-MEDIA", "AUTOMOBILE",
              "CHEF", "FINANCE", "APPAREL", "ENGINEERING", "ACCOUNTANT", "CONSTRUCTION", "PUBLIC-RELATIONS",
              "BANKING", "ARTS", "AVIATION"]


def download_resumes(cfg: ExperimentConfig) -> Path:
    raw = cfg.data_dir / "raw" / "resumes"
    out = raw / "Resume.csv"
    if out.exists():
        return out
    raw.mkdir(parents=True, exist_ok=True)
    comp = cfg.nlp["competition"]
    subprocess.run(["kaggle", "competitions", "download", "-c", comp, "-p", str(raw)], check=True)
    zp = raw / f"{comp}.zip"
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith("Resume.csv"))
        with z.open(name) as src, open(out, "wb") as dst:
            dst.write(src.read())
    (raw / "MANIFEST.json").write_text(json.dumps({"source": f"kaggle competitions download -c {comp}",
                                                   "zip_sha256": sha256(zp), "csv_sha256": sha256(out),
                                                   "member": name}, indent=2))
    zp.unlink()
    return out


def make_synthetic(cfg: ExperimentConfig) -> Path:
    """Textos sintéticos con el mismo esquema (solo perfil smoke)."""
    raw = cfg.data_dir / "raw" / "resumes"
    out = raw / "Resume.csv"
    if out.exists():
        return out
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    generic = ("managed team project client company city state skills experience summary education "
               "responsible developed customer service training reports communication").split()
    rows = []
    for i in range(cfg.nlp["n_docs"]):
        c = CATEGORIES[i % len(CATEGORIES)]
        kw = [c.lower().replace("-", " ")] + [f"{c[:3].lower()}{k}" for k in range(6)]
        n = int(rng.integers(40, 160))
        words = list(rng.choice(generic, n)) + list(rng.choice(kw, max(3, n // 8)))
        if rng.random() < 0.3:                                 # ruido: palabras de otra clase
            words += list(rng.choice([f"{CATEGORIES[(i + 1) % 24][:3].lower()}{k}" for k in range(6)], 4))
        rng.shuffle(words)
        title = f"{c.replace('-', ' ')} {'MANAGER' if rng.random() < .5 else 'SPECIALIST'}"
        text = f"         {title}       Summary  " + " ".join(words)
        rows.append(dict(ID=10_000_000 + i, Resume_str=text, Resume_html=f"<div>{text}</div>", Category=c))
    rows.append(dict(rows[0], ID=99_999_999))                   # duplicado exacto para probar la auditoría
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def load_resumes(cfg: ExperimentConfig) -> pd.DataFrame:
    path = make_synthetic(cfg) if cfg.nlp["source"] == "synthetic" else download_resumes(cfg)
    df = pd.read_csv(path)
    df = df.rename(columns={"Resume_str": "text", "Category": "label"})
    df["text"] = df["text"].fillna("").astype(str)
    return df[["ID", "text", "label"]]


# --------------------------------------------------------- preprocesado ----
_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MAIL = re.compile(r"\S+@\S+")
_NUM = re.compile(r"\d+")
_SYM = re.compile(r"[^a-z0-9\s]")


def clean_minimal(t: str) -> str:
    """Solo formato: espacios repetidos y bordes (para transformers, que ya manejan mayúsculas/puntuación)."""
    return _WS.sub(" ", t).strip()


def clean_standard(t: str) -> str:
    """Referencia de la guía: minúsculas, sin símbolos, sin stopwords."""
    t = _SYM.sub(" ", clean_minimal(t).lower())
    return " ".join(w for w in t.split() if w not in ENGLISH_STOP_WORDS)


_STEM = None


def clean_aggressive(t: str) -> str:
    """Estándar + sin URLs/correos/números + stemming de Porter (aproximación ligera a la lematización)."""
    global _STEM
    if _STEM is None:
        from nltk.stem import PorterStemmer
        _STEM = PorterStemmer()
    t = _NUM.sub(" ", _MAIL.sub(" ", _URL.sub(" ", t.lower())))
    return " ".join(_STEM.stem(w) for w in clean_standard(t).split() if len(w) > 1)


CLEANERS = {"minimal": clean_minimal, "standard": clean_standard, "aggressive": clean_aggressive}


def first_line_title(t: str) -> str:
    """Primera 'línea' de la hoja de vida (el cargo): candidato a atajo léxico."""
    parts = [p for p in re.split(r"\s{3,}", t.strip()) if p]
    return parts[0] if parts else ""


def mask_title(t: str) -> str:
    title = first_line_title(t)
    return t.replace(title, " ", 1) if title else t


# ------------------------------------------------------------ duplicados ---
def find_duplicates(texts: list[str], threshold: float = 0.8, num_perm: int = 128, k: int = 5):
    """Duplicados exactos (hash del texto normalizado) y casi-duplicados (MinHash + LSH, Jaccard ≥ umbral
    sobre 5-gramas de palabras). Devuelve (id de grupo por documento, lista de pares casi-duplicados)."""
    from datasketch import MinHash, MinHashLSH
    norm = [clean_standard(t) for t in texts]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mh = []
    for i, t in enumerate(norm):
        w = t.split()
        sh = {" ".join(w[j:j + k]) for j in range(max(1, len(w) - k + 1))}
        m = MinHash(num_perm=num_perm, seed=1)
        for s in sh:
            m.update(s.encode("utf8"))
        lsh.insert(str(i), m)
        mh.append(m)
    parent = list(range(len(texts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    pairs = []
    for i, m in enumerate(mh):
        for j in map(int, lsh.query(m)):
            if j > i:
                pairs.append((i, j, m.jaccard(mh[j]), norm[i] == norm[j]))
                parent[find(j)] = find(i)
    return np.array([find(i) for i in range(len(texts))]), pairs


def make_splits(df: pd.DataFrame, groups: np.ndarray, cfg: ExperimentConfig) -> dict:
    """70/15/15 estratificado, con los grupos de (casi-)duplicados en una sola partición + 5 folds externos."""
    y = df["label"].values
    idx = np.arange(len(df))
    n_test = round(1 / cfg.test_frac)
    a, test = next(StratifiedGroupKFold(n_test, shuffle=True, random_state=cfg.seed).split(idx, y, groups))
    n_val = round((1 - cfg.test_frac) / cfg.val_frac)
    b, c = next(StratifiedGroupKFold(n_val, shuffle=True, random_state=cfg.seed + 1).split(a, y[a], groups[a]))
    folds = [(a[i].tolist(), a[j].tolist()) for i, j in
             StratifiedGroupKFold(cfg.k_folds, shuffle=True, random_state=cfg.seed + 2).split(a, y[a], groups[a])]
    return {"train": a[b].tolist(), "val": a[c].tolist(), "test": test.tolist(), "folds": folds}


def assert_no_leak(groups: np.ndarray, sp: dict) -> None:
    g = {k: set(groups[sp[k]]) for k in ("train", "val", "test")}
    assert not (g["train"] & g["test"]) and not (g["val"] & g["test"]) and not (g["train"] & g["val"])


# -------------------------------------------------------- diversidad/legib. ---
def mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010), promedio de las dos direcciones."""
    def one(seq):
        factors, types, count = 0.0, set(), 0
        for w in seq:
            count += 1
            types.add(w)
            if len(types) / count <= ttr_threshold:
                factors += 1
                types, count = set(), 0
        if count:
            factors += (1 - len(types) / count) / (1 - ttr_threshold)
        return len(seq) / factors if factors else float(len(seq))
    return 0.5 * (one(tokens) + one(tokens[::-1])) if tokens else 0.0


def _syllables(w: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", w.lower())) - (1 if w.lower().endswith("e") else 0))


def flesch_reading_ease(t: str) -> float:
    sents = max(1, len(re.findall(r"[.!?]+", t)))
    words = re.findall(r"[A-Za-z]+", t)
    if not words:
        return 0.0
    return 206.835 - 1.015 * len(words) / sents - 84.6 * sum(map(_syllables, words)) / len(words)
