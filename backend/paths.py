from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RAW_CSV = DATA_RAW / "pc1.csv"
PROCESSED_CSV = DATA_PROCESSED / "modulos.csv"
MODEL_PATH = MODELS / "modelo.joblib"
METRICS_PATH = MODELS / "metrics.json"
META_PATH = MODELS / "meta.json"
