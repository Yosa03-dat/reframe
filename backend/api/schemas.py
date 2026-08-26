from pydantic import BaseModel
from typing import Optional

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    is_toxic: bool
    confidence: float
    flagged_segment: Optional[str] = None
    suggested_intervention: Optional[str] = None
