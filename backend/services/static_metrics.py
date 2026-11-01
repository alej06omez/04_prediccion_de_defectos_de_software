"""McCabe metrics computed deterministically with lizard."""
import os
import tempfile

import lizard


def extract_static_metrics(code: str) -> dict[str, int | float]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8") as source:
        source.write(code)
        path = source.name
    try:
        analysis = lizard.analyze_file(path)
        functions = analysis.function_list
        if not functions:
            return {"loc": int(analysis.nloc), "v_g": 1, "branchCount": 0}
        total_complexity = sum(fn.cyclomatic_complexity for fn in functions)
        complexity = total_complexity / len(functions)
        branches = max((total_complexity - len(functions)) * 2, 0)
        return {
            "loc": int(analysis.nloc),
            "v_g": complexity,
            "branchCount": branches,
        }
    finally:
        os.remove(path)
