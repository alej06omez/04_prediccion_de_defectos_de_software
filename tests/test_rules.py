from backend.rules import aplicar_reglas, clasificar_probabilidad


def test_umbrales():
    assert clasificar_probabilidad(0.1) == "bajo"
    assert clasificar_probabilidad(0.3) == "medio"
    assert clasificar_probabilidad(0.6) == "alto"


def test_modulo_critico_eleva_bajo():
    r = aplicar_reglas(0.1, loc=50, es_modulo_critico=True)
    assert r["nivel_segun_modelo"] == "bajo"
    assert r["nivel_riesgo"] == "medio"
    assert r["reglas_aplicadas"]


def test_baja_cobertura_y_loc_alto():
    r = aplicar_reglas(0.2, loc=250, cobertura_tests=0.1, es_modulo_critico=False)
    assert r["nivel_riesgo"] == "alto"
    assert r["origen_nivel"] == "reglas_negocio"
