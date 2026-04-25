"""
Pydantic models for structured LLM output across all agents.
Used with Strands structured_output_model for schema-enforced responses.
"""

from typing import Literal
from pydantic import BaseModel, Field


class QEResult(BaseModel):
    score: int = Field(ge=0, le=100)
    breakdown: dict = Field(default_factory=dict)
    feedback: str = ""
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class DebateResult(BaseModel):
    argument_for_a: str = ""
    argument_for_b: str = ""
    winner: Literal["A", "B"] = "A"
    reasoning: str = ""
    confidence: float = Field(default=0.5, ge=0.5, le=1.0)
