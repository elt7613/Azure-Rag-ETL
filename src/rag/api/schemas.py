"""Request and response shapes for the retrieval service."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(
        default=None,
        description="Groups turns into a conversation. Server-side history is "
        "keyed on it, so follow-ups resolve without the client resending "
        "the transcript.",
    )
    history: list[Message] = Field(
        default_factory=list,
        description="Optional client-supplied history, for stateless callers "
        "such as the evaluation harness.",
    )
    departments: list[str] | None = Field(
        default=None,
        description="Departments this caller may read. Omitted means the "
        "server decides from the caller's identity; if it cannot, the "
        "request retrieves nothing.",
    )
    use_cache: bool = True


class Citation(BaseModel):
    marker: int
    doc_id: str
    title: str = ""
    department: str = ""
    section: str = ""
    page: int = 0
    version: str = ""
    is_current: bool = True
    chunk_id: str = ""
    citation: str = ""


class ChatResponse(BaseModel):
    answer: str
    answered: bool
    abstained: bool = False
    clarification: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    conversation_id: str | None = None
    cached: bool = False
    # Everything needed to reconstruct why this answer looks the way it does:
    # what was searched for, what came back, what the sufficiency gate decided,
    # what verification found. Returned by default because "why did it say
    # that?" is the question this system is most often asked.
    diagnostics: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    dependencies: dict[str, str]


class StatsResponse(BaseModel):
    queries: int
    answered: int
    abstained: int
    clarified: int
    errors: int
    latency_ms: dict[str, float]
    cache: dict
    departments: list[str]
