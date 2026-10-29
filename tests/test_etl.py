from backend.etl import FEATURES, TARGET, clean, vectorize
from tests.conftest import tiny_pc1


def test_clean_assigns_module_ids():
    raw = tiny_pc1().drop(columns=["module_id"])
    out = clean(raw)
    assert list(out.columns[:2]) == ["module_id", "loc"]
    assert out[TARGET].isin([0, 1]).all()
    assert len(out) == len(raw)


def test_vectorize_imputes_missing():
    x, imputed = vectorize({"loc": 10}, fill={c: 1.5 for c in FEATURES})
    assert x.shape == (1, len(FEATURES))
    assert "v(g)" in imputed
    assert abs(x[0, 0] - 10) < 1e-9
