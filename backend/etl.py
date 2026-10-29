"""Ingesta y limpieza del dataset NASA PC1 (OpenML 1068)."""
from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CFG, DATA_PROCESSED, DATA_RAW, PROCESSED_CSV, RAW_CSV, ROOT

FEATURES = [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
]
TARGET = "defects"
FEATURE_ALIASES = {
    "lOCodeAndComment": "locCodeAndComment",
    "locCodeAndComment": "locCodeAndComment",
}
TARGET_ALIASES = ("defects", "Defects", "class", "bug", "defective")
PROMISE_ARFF = "http://promise.site.uottawa.ca/SERepository/datasets/pc1.arff"


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = str(col).strip()
        mapping[col] = FEATURE_ALIASES.get(key, key)
    return df.rename(columns=mapping)


def _target_column(df: pd.DataFrame) -> str:
    for name in TARGET_ALIASES:
        if name in df.columns:
            return name
    raise ValueError("El dataset no incluye columna de defectos (defects).")


def _to_binary(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    mapped = series.astype(str).str.strip().str.lower()
    true_vals = {"true", "yes", "1", "b'1'", "b'true'", "defective"}
    return mapped.isin(true_vals).astype(int)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_columns(df.copy())
    target_col = _target_column(df)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan metricas NASA: {missing}")
    out = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    out[TARGET] = _to_binary(df[target_col])
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if "module_id" in df.columns:
        ids = df.loc[out.index, "module_id"].astype(str)
    else:
        ids = [f"PC1-{i:04d}" for i in range(1, len(out) + 1)]
    out.insert(0, "module_id", list(ids))
    out = out.drop_duplicates(subset=["module_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("ETL dejo el dataset vacio.")
    return out


def dataset_hash(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def save_processed(df: pd.DataFrame, path: Path | None = None) -> Path:
    dest = path or PROCESSED_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def load_raw(path: Path | None = None) -> pd.DataFrame:
    src = path or RAW_CSV
    if not src.exists():
        raise FileNotFoundError(
            f"No hay dataset en {src}. Ejecute python descargar_dataset.py"
        )
    if src.suffix.lower() == ".arff":
        from scipy.io import arff

        data, _ = arff.loadarff(src)
        return pd.DataFrame(data)
    return pd.read_csv(src)


def load_processed(path: Path | None = None) -> pd.DataFrame:
    src = path or PROCESSED_CSV
    if src.exists():
        return pd.read_csv(src)
    cleaned = clean(load_raw())
    save_processed(cleaned, src)
    return cleaned


def medians(df: pd.DataFrame) -> dict[str, float]:
    return {c: float(df[c].median()) for c in FEATURES}


def vectorize(metricas: dict, fill: dict[str, float] | None = None) -> tuple[np.ndarray, list[str]]:
    fill = fill or {c: 0.0 for c in FEATURES}
    renamed = {FEATURE_ALIASES.get(k, k): v for k, v in metricas.items()}
    imputed: list[str] = []
    values: list[float] = []
    for col in FEATURES:
        if col in renamed and renamed[col] is not None and str(renamed[col]) != "":
            try:
                val = float(renamed[col])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Metrica no numerica: {col}") from exc
            if not np.isfinite(val):
                raise ValueError(f"Metrica no finita: {col}")
            values.append(val)
        else:
            values.append(float(fill[col]))
            imputed.append(col)
    return np.asarray([values], dtype=float), imputed


def descargar(destino: Path | None = None) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dest = destino or RAW_CSV
    try:
        from sklearn.datasets import fetch_openml

        cache = ROOT / "data" / "cache"
        ds = fetch_openml(
            data_id=int(CFG["openml_id"]),
            as_frame=True,
            parser="auto",
            data_home=str(cache),
        )
        frame = ds.frame
        frame.to_csv(dest, index=False)
        return dest
    except Exception:
        pass
    with urllib.request.urlopen(PROMISE_ARFF, timeout=30) as resp:
        raw = resp.read()
    from scipy.io import arff

    data, _ = arff.loadarff(io.BytesIO(raw))
    pd.DataFrame(data).to_csv(dest, index=False)
    return dest


def preparar(raw_path: Path | None = None) -> pd.DataFrame:
    df = clean(load_raw(raw_path) if raw_path else load_raw())
    save_processed(df)
    return df
