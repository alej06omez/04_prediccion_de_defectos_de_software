"""Entrenamiento versionado y prediccion por modulo. No aplica reglas de negocio."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .etl import FEATURES, TARGET, dataset_hash, load_processed, medians, save_processed, vectorize
from .paths import META_PATH, METRICS_PATH, MODEL_PATH, MODELS
from .rules import aplicar_reglas


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_bundle() -> dict:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Modelo no entrenado. POST /train o python -m backend.ml")
    return {
        "modelo": joblib.load(MODEL_PATH),
        "meta": json.loads(META_PATH.read_text(encoding="utf-8")),
        "metrics": json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {},
    }


def train(df=None) -> dict:
    data = df if df is not None else load_processed()
    if TARGET not in data.columns:
        raise ValueError("El dataset procesado no tiene etiqueta defects.")
    x = data[FEATURES].astype(float)
    y = data[TARGET].astype(int)
    if y.nunique() < 2:
        raise ValueError("Se necesitan ambas clases (defecto / no defecto) para entrenar.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred).tolist()
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "matriz_confusion": {"tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]},
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "tasa_defectos": float(y.mean()),
        "umbral_clase": 0.5,
    }
    clf = model.named_steps["logisticregression"]
    coef = {FEATURES[i]: float(clf.coef_[0][i]) for i in range(len(FEATURES))}
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "version_modelo": version,
        "algoritmo": "StandardScaler + LogisticRegression(class_weight=balanced)",
        "features": FEATURES,
        "dataset_hash": dataset_hash(data),
        "n_filas": int(len(data)),
        "entrenado_en": _now(),
        "imputacion": medians(data),
        "coeficientes": coef,
        "nota": "El modelo estima P(defecto). El semaforo de riesgo lo aplican reglas de negocio.",
    }
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_processed(data)
    return {"meta": meta, "metrics": metrics}


def explicar(modelo, x: pd.DataFrame, top: int = 3) -> list[dict]:
    scaler = modelo.named_steps["standardscaler"]
    clf = modelo.named_steps["logisticregression"]
    xs = scaler.transform(x)[0]
    contrib = xs * clf.coef_[0]
    values = x.to_numpy()[0]
    order = np.argsort(np.abs(contrib))[::-1][:top]
    out = []
    for i in order:
        out.append(
            {
                "metrica": FEATURES[int(i)],
                "valor": float(values[i]),
                "contribucion": float(contrib[i]),
                "efecto": "aumenta_riesgo" if contrib[i] > 0 else "reduce_riesgo",
            }
        )
    return out


def predict(payload: dict) -> dict:
    bundle = _load_bundle()
    modelo = bundle["modelo"]
    meta = bundle["meta"]
    metricas = payload.get("metricas") or {}
    if not isinstance(metricas, dict) or not metricas:
        raise ValueError("Se requiere metricas: objeto con nombres NASA (loc, v(g), ...).")
    x, imputed = vectorize(metricas, meta.get("imputacion"))
    x_df = pd.DataFrame(x, columns=FEATURES)
    proba = float(modelo.predict_proba(x_df)[0, 1])
    loc = float(x_df.iloc[0]["loc"])
    reglas = aplicar_reglas(
        proba,
        loc=loc,
        cobertura_tests=payload.get("cobertura_tests"),
        es_modulo_critico=bool(payload.get("es_modulo_critico", False)),
    )
    return {
        "module_id": str(payload.get("module_id", "sin-id")),
        "probabilidad_defecto": round(proba, 4),
        "prediccion_clase": int(proba >= 0.5),
        "explicacion": explicar(modelo, x_df),
        "metricas_imputadas": imputed,
        "version_modelo": meta["version_modelo"],
        "dataset_hash": meta["dataset_hash"],
        "nota": "IA: probabilidad. Negocio: nivel_riesgo y reglas_aplicadas.",
        **reglas,
    }


def ranking(limit: int = 40) -> list[dict]:
    bundle = _load_bundle()
    data = load_processed()
    proba = bundle["modelo"].predict_proba(data[FEATURES].astype(float))[:, 1]
    rows = []
    for i, p in enumerate(proba):
        loc = float(data.iloc[i]["loc"])
        reglas = aplicar_reglas(float(p), loc=loc)
        rows.append(
            {
                "module_id": str(data.iloc[i]["module_id"]),
                "probabilidad_defecto": round(float(p), 4),
                "defecto_real": int(data.iloc[i][TARGET]) if TARGET in data.columns else None,
                "loc": loc,
                "v(g)": float(data.iloc[i]["v(g)"]),
                **reglas,
            }
        )
    rows.sort(key=lambda r: r["probabilidad_defecto"], reverse=True)
    return rows[: max(1, min(limit, 200))]


def estado_modelo() -> dict:
    bundle = _load_bundle()
    return {**bundle["meta"], "metrics": bundle["metrics"]}


if __name__ == "__main__":
    from .etl import preparar

    preparar()
    info = train()
    print(json.dumps({"version": info["meta"]["version_modelo"], "metrics": info["metrics"]}, indent=2))
