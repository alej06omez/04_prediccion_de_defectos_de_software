from backend.services.halstead_calculator import compute_halstead, estimate_metrics_deterministic
from backend.services.static_metrics import extract_static_metrics


SAMPLE_C = """#include <stdio.h>
int main(void) {
    int x = 1;
    if (x < 2) x = x + 1;
    return x;
}
"""


def test_halstead_token_counts_and_formulas_are_deterministic():
    first = compute_halstead(SAMPLE_C)
    second = compute_halstead(SAMPLE_C)
    assert first == second
    assert first["uniq_Op"] > 0
    assert first["uniq_Opnd"] > 0
    assert first["total_Op"] >= first["uniq_Op"]
    assert first["total_Opnd"] >= first["uniq_Opnd"]


def test_full_metric_contract_is_identical_between_runs():
    static = extract_static_metrics(SAMPLE_C)
    first = estimate_metrics_deterministic(SAMPLE_C, "sample", static).model_dump(by_alias=True)
    second = estimate_metrics_deterministic(SAMPLE_C, "sample", static).model_dump(by_alias=True)
    assert first == second
    assert first["ev(g)"] == first["iv(g)"] == first["v(g)"]


def test_preprocessor_lines_are_not_counted_as_halstead_tokens():
    without_include = SAMPLE_C.replace("#include <stdio.h>\n", "")
    assert compute_halstead(SAMPLE_C) == compute_halstead(without_include)


def test_static_complexity_is_averaged_per_function():
    code = "int a(void) { return 1; }\nint b(void) { if (1) return 2; return 0; }"
    metrics = extract_static_metrics(code)
    assert metrics["v_g"] == 1.5
