import re


def detect_language(code: str) -> tuple[str, bool]:
    """Small deterministic guard for pasted C source (not a full parser)."""
    if not code.strip():
        return "empty", False
    c_markers = ("#include", "int main", "void ", "struct ", "typedef ")
    has_c_marker = any(marker in code for marker in c_markers) or bool(
        re.search(r"\b(?:int|char|float|double|long|short)\s+\w+\s*\(", code)
    )
    looks_like_other_language = bool(
        re.search(r"\b(?:def|import|function|const|let|var)\b", code)
    ) and not has_c_marker
    if looks_like_other_language:
        return "other", False
    return ("c", True) if has_c_marker else ("unknown", False)
