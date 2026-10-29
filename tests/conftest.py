import pandas as pd
import pytest

from backend.etl import FEATURES, TARGET


def tiny_pc1(n=40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        defect = i % 4 == 0
        loc = 40 + i * 8 + (120 if defect else 0)
        vg = 2 + i % 7 + (10 if defect else 0)
        row = {c: float((i + 1) * (k + 1) % 50 + 1) for k, c in enumerate(FEATURES)}
        row["loc"] = float(loc)
        row["v(g)"] = float(vg)
        row["branchCount"] = float(vg * 2)
        row["lOComment"] = float(0 if defect else 12)
        row[TARGET] = int(defect)
        row["module_id"] = f"M-{i:03d}"
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df(tmp_path, monkeypatch):
    df = tiny_pc1()
    processed = tmp_path / "modulos.csv"
    df.to_csv(processed, index=False)
    monkeypatch.setattr("backend.etl.PROCESSED_CSV", processed)
    monkeypatch.setattr("backend.ml.MODEL_PATH", tmp_path / "modelo.joblib")
    monkeypatch.setattr("backend.ml.METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr("backend.ml.META_PATH", tmp_path / "meta.json")
    monkeypatch.setattr("backend.ml.MODELS", tmp_path)
    return df
