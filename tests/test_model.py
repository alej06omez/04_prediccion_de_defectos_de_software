from backend.ml import predict, train
from backend.rules import aplicar_reglas, clasificar_probabilidad


def test_train_and_predict_module(sample_df):
    info = train(sample_df)
    assert "roc_auc" in info["metrics"]
    assert info["meta"]["version_modelo"]
    out = predict(
        {
            "module_id": "auth-service",
            "metricas": {k: 10.0 for k in info["meta"]["features"]},
        }
    )
    assert out["module_id"] == "auth-service"
    assert 0 <= out["probabilidad_defecto"] <= 1
    assert out["explicacion"]
    assert "nivel_riesgo" in out


def test_predict_requires_named_metrics(sample_df):
    train(sample_df)
    try:
        predict({"module_id": "x", "valores": [1, 2, 3]})
        assert False, "debia fallar"
    except ValueError:
        pass
