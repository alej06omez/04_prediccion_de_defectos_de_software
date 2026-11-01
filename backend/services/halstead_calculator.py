"""Deterministic C token counts and Halstead formulas."""
import math
import re

from pycparser.c_lexer import CLexer

from backend.schemas.code_metrics import CodeMetrics

KEYWORDS = {
    "AUTO", "BREAK", "CASE", "CHAR", "CONST", "CONTINUE", "DEFAULT", "DO",
    "DOUBLE", "ELSE", "ENUM", "EXTERN", "FLOAT", "FOR", "GOTO", "IF", "INLINE",
    "INT", "LONG", "REGISTER", "RESTRICT", "RETURN", "SHORT", "SIGNED", "SIZEOF",
    "STATIC", "STRUCT", "SWITCH", "TYPEDEF", "UNION", "UNSIGNED", "VOID", "VOLATILE",
    "WHILE", "_BOOL", "_COMPLEX", "_IMAGINARY",
}
OPERATORS = {
    "PLUS", "MINUS", "TIMES", "DIVIDE", "MOD", "OR", "AND", "NOT", "XOR", "LSHIFT",
    "RSHIFT", "LOR", "LAND", "LNOT", "LT", "LE", "GT", "GE", "EQ", "NE", "EQUALS",
    "TIMESEQUAL", "DIVEQUAL", "MODEQUAL", "PLUSEQUAL", "MINUSEQUAL", "LSHIFTEQUAL",
    "RSHIFTEQUAL", "ANDEQUAL", "XOREQUAL", "OREQUAL", "PLUSPLUS", "MINUSMINUS", "ARROW",
    "CONDOP", "LPAREN", "RPAREN", "LBRACKET", "RBRACKET", "LBRACE", "RBRACE", "COMMA",
    "PERIOD", "SEMI", "COLON", "ELLIPSIS",
}
OPERANDS = {
    "ID", "TYPEID", "INT_CONST_DEC", "INT_CONST_OCT", "INT_CONST_HEX", "INT_CONST_BIN",
    "INT_CONST_CHAR", "FLOAT_CONST", "HEX_FLOAT_CONST", "CHAR_CONST", "WCHAR_CONST",
    "U8CHAR_CONST", "U16CHAR_CONST", "U32CHAR_CONST", "STRING_LITERAL", "WSTRING_LITERAL",
    "U8STRING_LITERAL", "U16STRING_LITERAL", "U32STRING_LITERAL",
}


def _without_preprocessor(code: str) -> str:
    # Remove complete directives, including backslash-continued lines, before lexing.
    kept = []
    skipping = False
    for line in code.splitlines(keepends=True):
        if not skipping and re.match(r"^\s*#", line):
            skipping = line.rstrip("\r\n").endswith("\\")
            kept.append("\n" if line.endswith(("\n", "\r")) else "")
        elif skipping:
            skipping = line.rstrip("\r\n").endswith("\\")
            kept.append("\n" if line.endswith(("\n", "\r")) else "")
        else:
            kept.append(line)
    return "".join(kept)


def _tokenize(code: str) -> list[tuple[str, str]]:
    def ignore_error(*_args):
        return None

    lexer = CLexer(
        error_func=ignore_error,
        on_lbrace_func=lambda: None,
        on_rbrace_func=lambda: None,
        type_lookup_func=lambda _name: False,
    )
    lexer.build()
    lexer.input(_without_preprocessor(code))
    result = []
    while token := lexer.token():
        result.append((token.type, token.value))
    return result


def compute_halstead(code: str) -> dict[str, int | float]:
    unique_operators: set[str] = set()
    unique_operands: set[str] = set()
    total_operators = total_operands = 0
    for kind, value in _tokenize(code):
        if kind in KEYWORDS or kind in OPERATORS:
            unique_operators.add(value)
            total_operators += 1
        elif kind in OPERANDS:
            unique_operands.add(value)
            total_operands += 1

    n1, n2 = len(unique_operators), len(unique_operands)
    vocabulary = n1 + n2
    length = total_operators + total_operands
    volume = length * math.log2(vocabulary) if vocabulary > 1 else 0.0
    difficulty = (n1 / 2) * (total_operands / n2) if n2 else 0.0
    effort = difficulty * volume
    level = volume / effort if effort else 0.0
    return {
        "n": vocabulary,
        "v": round(volume, 2),
        "l": round(level, 4),
        "d": round(difficulty, 2),
        "i": round(volume * level, 2),
        "e": round(effort, 2),
        "b": round(volume / 3000, 4),
        "t": round(effort / 18, 2),
        "uniq_Op": n1,
        "uniq_Opnd": n2,
        "total_Op": total_operators,
        "total_Opnd": total_operands,
    }


def compute_line_metrics(code: str) -> dict[str, int]:
    lines = code.splitlines()
    blank = comments = code_and_comment = code_lines = 0
    in_block_comment = False
    for raw in lines:
        line = raw.strip()
        if not line:
            blank += 1
        elif in_block_comment:
            comments += 1
            if "*/" in line:
                in_block_comment = False
        elif line.startswith("//") or line.startswith("/*"):
            comments += 1
            in_block_comment = line.startswith("/*") and "*/" not in line
        elif "//" in line or "/*" in line:
            code_and_comment += 1
            in_block_comment = "/*" in line and "*/" not in line
        else:
            code_lines += 1
    return {
        "lOCode": code_lines,
        "lOComment": comments,
        "lOBlank": blank,
        "locCodeAndComment": code_and_comment,
    }


def estimate_metrics_deterministic(code: str, module_id: str, mccabe_metrics: dict) -> CodeMetrics:
    vg = mccabe_metrics["v_g"]
    return CodeMetrics(
        module_id=module_id,
        loc=mccabe_metrics["loc"],
        **{"v(g)": vg, "ev(g)": vg, "iv(g)": vg},
        branchCount=mccabe_metrics["branchCount"],
        **compute_halstead(code),
        **compute_line_metrics(code),
    )
