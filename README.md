# Predicción de defectos de software

**Curso:** Inteligencia Artificial  
**Docente:** Julian Rene Muñoz Burbano

Sistema que estima **P(defecto)** por módulo con métricas NASA PC1 (McCabe/Halstead) y aplica **reglas de negocio** para el semáforo de riesgo. Sirve para priorizar revisión y pruebas, no para sustituirlas.

## Qué decide la IA y qué decide el negocio

| IA (modelo) | Reglas deterministas |
|---|---|
| Probabilidad de defecto y explicación por coeficientes | Umbrales bajo/medio/alto (0.3 / 0.6) |
| Entrenamiento versionado (hash del dataset, ROC-AUC, F1) | Módulo crítico no puede quedar en bajo |
| Ranking por probabilidad | Cobertura baja + LOC alto → riesgo alto |

## Instalación Windows / Anaconda

```bat
conda create -n IA_C1_P04 python=3.11 -y
conda activate IA_C1_P04
python -m pip install -r requirements.txt
copy .env.example .env
python descargar_dataset.py
python -m backend.ml
```

Instale MongoDB Community Server en localhost:27017, o use Docker Compose.

## Ejecución

```bat
uvicorn backend.api:app --reload --port 8000
streamlit run frontend/app.py
```

Frontend: http://localhost:8501 · API: http://localhost:8000/docs

Roles en el sidebar: **operativo** (predicción), **analista** (métricas del modelo), **admin** (reentrenar). Si define `API_KEY` en `.env`, envíela en `X-API-Key`.

## Análisis determinista de código C

`POST /api/analyze-code` recibe `{"code": "...", "module_id": "..."}`. El endpoint valida que la entrada parezca C, obtiene `loc`, `v(g)` y `branchCount` con `lizard`, y calcula Halstead a partir de tokens de C con `pycparser` y fórmulas fijas. El resultado y el hash SHA-256 del código se guardan en MongoDB; no se envía el código a un LLM ni se requiere una API key de proveedor de LLM.

`ev(g)` e `iv(g)` se devuelven iguales a `v(g)` como valores `estimated`. Su cálculo exacto requiere construir y analizar el grafo de flujo de control completo; el endpoint lo declara en `warnings` y en la descripción OpenAPI. La clasificación de operandos y operadores sigue una convención léxica determinista de C, que puede diferir de las convenciones usadas al generar NASA PC1.

## Docker

```bat
docker compose up --build
```

## Pruebas

```bat
python -m pytest -q
```

## Dataset

- NASA PC1, OpenML [1068](https://www.openml.org/d/1068) / PROMISE.
- Crudo: `data/raw/pc1.csv`. Procesado: `data/processed/modulos.csv`.
- Artefactos: `models/modelo.joblib`, `models/metrics.json`, `models/meta.json`.
