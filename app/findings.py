from typing import Literal
from pydantic import BaseModel


class Finding(BaseModel):
    id: str
    category: Literal["metadata", "provenance", "pixel", "audio", "text", "tracking"]
    name: str
    severity: Literal["info", "low", "medium", "high"]
    confidence: float
    evidence: str
    removable: bool
    removal: str
