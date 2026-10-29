"""Reglas de negocio deterministas. El modelo no decide semaforos ni bloqueos."""
from __future__ import annotations

from .paths import CFG

UMBRALES = CFG["umbrales_riesgo"]


def clasificar_probabilidad(probabilidad: float) -> str:
    if probabilidad >= float(UMBRALES["alto"]):
        return "alto"
    if probabilidad >= float(UMBRALES["medio"]):
        return "medio"
    return "bajo"


def aplicar_reglas(
    probabilidad: float,
    loc: float | None = None,
    cobertura_tests: float | None = None,
    es_modulo_critico: bool = False,
) -> dict:
    nivel_ia = clasificar_probabilidad(probabilidad)
    nivel = nivel_ia
    razones: list[str] = []

    if es_modulo_critico and nivel == "bajo":
        nivel = "medio"
        razones.append("modulo critico: el riesgo bajo se eleva a medio")

    cobertura_min = float(CFG["cobertura_minima"])
    loc_limite = float(CFG["loc_sin_cobertura_alto"])
    if cobertura_tests is not None and loc is not None:
        if cobertura_tests < cobertura_min and loc >= loc_limite:
            nivel = "alto"
            razones.append(
                f"regla: cobertura {cobertura_tests:.0%} < {cobertura_min:.0%} y LOC {loc:.0f} >= {loc_limite:.0f}"
            )

    return {
        "nivel_riesgo": nivel,
        "nivel_segun_modelo": nivel_ia,
        "reglas_aplicadas": razones,
        "origen_nivel": "reglas_negocio" if razones else "umbral_sobre_probabilidad",
        "umbrales": UMBRALES,
    }
