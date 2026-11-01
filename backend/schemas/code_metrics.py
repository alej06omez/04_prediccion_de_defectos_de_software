from pydantic import BaseModel, ConfigDict, Field


class CodeMetrics(BaseModel):
    """NASA PC1 feature contract (21 metrics plus module_id)."""

    model_config = ConfigDict(populate_by_name=True)

    module_id: str
    loc: float
    v_g: float = Field(alias="v(g)")
    ev_g: float = Field(alias="ev(g)")
    iv_g: float = Field(alias="iv(g)")
    n: float
    v: float
    l: float
    d: float
    i: float
    e: float
    b: float
    t: float
    lOCode: float
    lOComment: float
    lOBlank: float
    locCodeAndComment: float
    uniq_Op: float
    uniq_Opnd: float
    total_Op: float
    total_Opnd: float
    branchCount: float


class AnalyzeCodeRequest(BaseModel):
    code: str = Field(..., min_length=1)
    module_id: str = Field(..., min_length=1, max_length=120)
