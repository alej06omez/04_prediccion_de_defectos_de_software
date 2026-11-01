import logging
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from .etl import FEATURES
from .ml import estado_modelo, predict, ranking, train
from .paths import CFG, ROOT
from .schemas.code_metrics import AnalyzeCodeRequest
from .services.halstead_calculator import estimate_metrics_deterministic
from .services.static_metrics import extract_static_metrics
from .validators.language_detector import detect_language

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sdp")

app = FastAPI(
    title=CFG["titulo"],
    description=(
        "Estima P(defecto) por modulo a partir de metricas NASA PC1. "
        "El semaforo de riesgo es regla de negocio, no salida cruda del modelo. "
        "POST /api/analyze-code calcula métricas C de forma determinista. "
        "ev(g) e iv(g) se devuelven iguales a v(g) como estimaciones: "
        "su cálculo exacto requiere el grafo de flujo de control. "
        "Roles: operativo (prediccion), analista (metricas/explicacion), admin (entrenamiento)."
    ),
    version="2.0.0",
)


class ModuloEntrada(BaseModel):
    module_id: str = Field(..., min_length=1, max_length=120)
    metricas: dict[str, float]
    cobertura_tests: float | None = Field(default=None, ge=0, le=1)
    es_modulo_critico: bool = False
    descripcion: str = Field(default="", max_length=500)


class LoteEntrada(BaseModel):
    modulos: list[ModuloEntrada] = Field(..., min_length=1, max_length=100)


def collection():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    return client[os.getenv("MONGO_DB", CFG["db"])]["predicciones"]


def persist(entrada: dict, resultado: dict) -> str | None:
    doc = {
        "entrada": entrada,
        "resultado": resultado,
        "fecha": datetime.now(timezone.utc),
        "version_modelo": resultado.get("version_modelo"),
    }
    try:
        saved = collection().insert_one(doc)
        return str(saved.inserted_id)
    except PyMongoError as exc:
        log.error("MongoDB no disponible: %s", exc)
        raise HTTPException(503, "MongoDB no disponible: " + str(exc)) from exc


def require_admin(x_api_key: str | None, x_role: str | None) -> None:
    expected = os.getenv("API_KEY", "").strip()
    role = (x_role or "operativo").lower()
    if expected:
        if x_api_key != expected:
            raise HTTPException(401, "API key invalida")
        return
    if role != "admin":
        raise HTTPException(403, "El entrenamiento esta reservado al rol administrador")


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        ms = (time.perf_counter() - start) * 1000
        log.exception("%s %s fallo tras %.1fms", request.method, request.url.path, ms)
        raise
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/health")
def health():
    mongo_ok = True
    try:
        collection()
    except Exception:
        mongo_ok = False
    modelo_ok = Path(ROOT / "models" / "modelo.joblib").exists()
    return {
        "estado": "api_activa",
        "proyecto": CFG["titulo"],
        "mongodb": mongo_ok,
        "modelo_entrenado": modelo_ok,
    }


@app.get("/schema")
def schema():
    return {
        "features": FEATURES,
        "tipo_modelo": CFG["tipo_modelo"],
        "umbrales_riesgo": CFG["umbrales_riesgo"],
        "ia": "LogisticRegression sobre metricas McCabe/Halstead -> probabilidad de defecto",
        "negocio": "umbrales bajo/medio/alto, modulo critico y cobertura vs LOC",
    }


@app.get("/model")
def model_info():
    try:
        return estado_modelo()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/train")
def run_train(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_role: str | None = Header(default=None, alias="X-Role"),
):
    require_admin(x_api_key, x_role)
    try:
        from .etl import preparar

        preparar()
        return train()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/predict")
def run_predict(item: ModuloEntrada):
    try:
        resultado = predict(item.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    resultado["id"] = persist(item.model_dump(exclude_none=True), resultado)
    return resultado


@app.post("/predict/lote")
def run_lote(item: LoteEntrada):
    salidas = []
    for modulo in item.modulos:
        try:
            resultado = predict(modulo.model_dump())
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, f"{modulo.module_id}: {exc}") from exc
        resultado["id"] = persist(modulo.model_dump(exclude_none=True), resultado)
        salidas.append(resultado)
    salidas.sort(key=lambda r: r["probabilidad_defecto"], reverse=True)
    return salidas


@app.post(
    "/api/analyze-code",
    summary="Calcular métricas deterministas de código C",
    description=(
        "Calcula McCabe/LOC con lizard y Halstead mediante tokenización C y fórmulas fijas. "
        "ev(g) e iv(g) se publican iguales a v(g), marcados como estimated porque "
        "su valor exacto requiere analizar el grafo de flujo de control completo."
    ),
)
def analyze_code(item: AnalyzeCodeRequest):
    language, is_c = detect_language(item.code)
    if not is_c:
        raise HTTPException(422, f"El código no parece ser C (detectado: {language}). Solo se soporta C.")
    try:
        static = extract_static_metrics(item.code)
        metrics = estimate_metrics_deterministic(item.code, item.module_id, static)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, f"No se pudieron calcular las métricas: {exc}") from exc

    unverified = {"ev(g)", "iv(g)"}
    result = {
        "metrics": metrics.model_dump(by_alias=True),
        "language_detected": language,
        "sources": [
            {
                "field": field,
                "source": "derived_from_v(g)" if field in unverified else "deterministic_static_analysis",
                "confidence": "estimated" if field in unverified else "exact",
            }
            for field in metrics.model_dump(by_alias=True)
            if field != "module_id"
        ],
        "warnings": [
            "ev(g) e iv(g) se devuelven iguales a v(g) y marcadas como estimated; "
            "su cálculo exacto requiere el grafo de flujo de control completo."
        ],
        "method": "deterministic: lizard + pycparser lexer + Halstead formulas",
    }
    result["id"] = persist(
        {"module_id": item.module_id, "code_sha256": hashlib.sha256(item.code.encode("utf-8")).hexdigest()},
        result,
    )
    return result


@app.get("/ranking")
def run_ranking(limit: int = 40):
    if not 1 <= limit <= 200:
        raise HTTPException(422, "limit debe estar entre 1 y 200")
    try:
        return ranking(limit)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/records")
def records(limit: int = 20):
    if not 1 <= limit <= 200:
        raise HTTPException(422, "limit debe estar entre 1 y 200")
    try:
        docs = list(collection().find({}, sort=[("fecha", -1)], limit=limit))
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            doc["fecha"] = doc["fecha"].isoformat()
        return docs
    except PyMongoError as exc:
        raise HTTPException(503, "MongoDB no disponible: " + str(exc)) from exc
