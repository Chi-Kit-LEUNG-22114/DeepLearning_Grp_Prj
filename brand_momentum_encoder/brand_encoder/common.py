from __future__ import annotations
import hashlib, json, random, re
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import yaml

NULL_TEXT = {"", "nan", "none", "null", "deleted", "removed", "[deleted]", "[removed]"}

def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f) or {}

def ensure_dir(path: str | Path) -> Path:
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p

def stable_id(*parts: Any) -> str:
    raw="|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

def file_sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception: pass

def read_any(path: Path) -> pd.DataFrame:
    s=path.suffix.lower()
    if s==".csv": return pd.read_csv(path)
    if s==".tsv": return pd.read_csv(path, sep="\t")
    if s in {".jsonl",".ndjson"}: return pd.read_json(path, lines=True)
    if s==".json": return pd.read_json(path)
    if s==".parquet": return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")

def write_json(path: Path, obj: Any)->None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def normalize_missing_text(value: Any) -> str:
    if value is None: return ""
    s=str(value).strip()
    return "" if s.lower() in NULL_TEXT else s

def week_start(values: pd.Series, timezone: str) -> pd.Series:
    dt=pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(timezone).dt.tz_localize(None)
    return (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()

def numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df: df[c]=pd.to_numeric(df[c], errors="coerce")
    return df

def select_device(name: str):
    import torch
    if name=="auto":
        if torch.cuda.is_available(): return "cuda"
        if getattr(torch.backends,"mps",None) is not None and torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return name
