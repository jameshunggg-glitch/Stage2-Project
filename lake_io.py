# lake_io.py
from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _choose_fmt(path_no_ext: Path, prefer="parquet") -> tuple[Path, str]:
    """
    Prefer parquet; fallback to pickle if parquet engine not available.
    """
    if prefer == "parquet":
        return path_no_ext.with_suffix(".parquet"), "parquet"
    if prefer == "pickle":
        return path_no_ext.with_suffix(".pkl"), "pickle"
    if prefer == "csv":
        return path_no_ext.with_suffix(".csv"), "csv"
    return path_no_ext.with_suffix(".parquet"), "parquet"

def save_df(df: pd.DataFrame, out_dir: Path, name: str, *, prefer="parquet", meta: dict | None = None) -> Path:
    out_dir = ensure_dir(out_dir)
    base = out_dir / name
    path, fmt = _choose_fmt(base, prefer=prefer)

    if fmt == "parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception:
            # fallback
            path = base.with_suffix(".pkl")
            df.to_pickle(path)
            fmt = "pickle"
    elif fmt == "pickle":
        df.to_pickle(path)
    elif fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_all = {
        "name": name,
        "saved_at": _now_tag(),
        "format": fmt,
        "rows": int(len(df)),
        "cols": list(df.columns),
    }
    if meta:
        meta_all.update(meta)

    meta_path.write_text(json.dumps(meta_all, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def load_df(out_dir: Path, name: str) -> pd.DataFrame | None:
    """
    Load name.parquet or name.pkl or name.csv if exists, else None.
    """
    out_dir = Path(out_dir)
    cand = [
        out_dir / f"{name}.parquet",
        out_dir / f"{name}.pkl",
        out_dir / f"{name}.csv",
    ]
    for p in cand:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            if p.suffix == ".pkl":
                return pd.read_pickle(p)
            if p.suffix == ".csv":
                return pd.read_csv(p)
    return None
