"""Registro de modelos mínimo basado en archivos, con etapas y reversión.

    python -m app.registry register <dir> --name text --version 1.0.0 --metrics '{"f1": 0.8}'
    python -m app.registry promote text 1.0.0 production
    python -m app.registry rollback text

Cada versión vive en ``models_registry/<name>/<version>/``; ``registry.json`` guarda el historial de etapas.
MLflow (cuando está instalado) registra los mismos artefactos con alias 'staging'/'production'.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "models_registry"
INDEX = ROOT / "registry.json"


def _load() -> dict:
    return json.loads(INDEX.read_text()) if INDEX.exists() else {}


def _save(d: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(d, indent=1))


def register(src: Path, name: str, version: str, metrics: dict) -> Path:
    dst = ROOT / name / version
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    d = _load()
    d.setdefault(name, {"versions": {}, "history": []})
    d[name]["versions"][version] = {"metrics": metrics, "registered": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "stage": "staging"}
    _save(d)
    return dst


def promote(name: str, version: str, stage: str = "production") -> None:
    d = _load()
    for v, meta in d[name]["versions"].items():
        if stage == "production" and meta["stage"] == "production":
            meta["stage"] = "archived"
            d[name]["history"].append(v)
    d[name]["versions"][version]["stage"] = stage
    _save(d)


def rollback(name: str) -> str:
    d = _load()
    prev = d[name]["history"].pop()
    for v, meta in d[name]["versions"].items():
        if meta["stage"] == "production":
            meta["stage"] = "archived"
    d[name]["versions"][prev]["stage"] = "production"
    _save(d)
    return prev


def production(name: str) -> str | None:
    d = _load().get(name, {}).get("versions", {})
    return next((v for v, m in d.items() if m["stage"] == "production"), None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("src")
    r.add_argument("--name", required=True)
    r.add_argument("--version", required=True)
    r.add_argument("--metrics", default="{}")
    p = sub.add_parser("promote")
    p.add_argument("name")
    p.add_argument("version")
    p.add_argument("stage", nargs="?", default="production")
    b = sub.add_parser("rollback")
    b.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "register":
        print(register(Path(a.src), a.name, a.version, json.loads(a.metrics)))
    elif a.cmd == "promote":
        promote(a.name, a.version, a.stage)
    else:
        print("producción ahora:", rollback(a.name))
