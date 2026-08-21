"""The HTTP surface, exercised against the live pipeline."""
from __future__ import annotations

import httpx
import pytest

from tests.conftest import azure_configured

from rag.api.security import resolve_scope
from rag.config import get_settings

live = pytest.mark.skipif(
    not azure_configured(), reason="Azure credentials not configured"
)

HEADER = "X-User-Departments"


# ---------------- scope resolution (pure) ----------------


def test_no_identity_grants_nothing_when_no_default_is_configured(monkeypatch):
    from rag import api

    settings = get_settings()
    monkeypatch.setattr(settings, "api_default_departments", [], raising=False)
    assert resolve_scope(None, None) == []
    # A client cannot grant itself access by asking.
    assert resolve_scope(["HR"], None) == []


def test_header_grants_scope(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "api_default_departments", [], raising=False)
    assert resolve_scope(None, "HR,finance") == ["HR", "finance"]


def test_body_can_only_narrow_what_the_header_granted(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "api_default_departments", [], raising=False)
    # Narrowing is allowed.
    assert resolve_scope(["HR"], "HR,finance") == ["HR"]
    # Widening is not — this is the privilege-escalation path.
    assert resolve_scope(["HR", "legal"], "HR") == ["HR"]
    assert resolve_scope(["legal"], "HR") == []


def test_unknown_departments_are_dropped(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "api_default_departments", [], raising=False)
    assert resolve_scope(None, "HR,marketing,finance") == ["HR", "finance"]


def test_scope_matching_is_case_insensitive_but_canonical(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "api_default_departments", [], raising=False)
    assert resolve_scope(None, "hr,FINANCE") == ["HR", "finance"]


# ---------------- live HTTP ----------------


@pytest.fixture
async def client():
    from rag.api.main import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=120.0
    ) as http_client:
        async with app.router.lifespan_context(app):
            yield http_client


@live
async def test_departments_endpoint_reflects_configuration(client):
    response = await client.get("/departments")
    assert response.status_code == 200
    names = [d["name"] for d in response.json()["departments"]]
    assert names == get_settings().departments


@live
async def test_health_probes_each_dependency(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["dependencies"]["azure_search"] == "ok"
    assert body["dependencies"]["azure_openai"] == "ok"


@live
async def test_chat_answers_with_citations(client, facts):
    response = await client.post(
        "/chat",
        json={"message": "How many days of paid sick leave do employees get?"},
        headers={HEADER: "HR"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert facts.sick_leave_days in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["doc_id"]
    assert body["conversation_id"]


@live
async def test_chat_without_scope_is_refused_with_a_reason(client):
    """403 rather than an empty answer: an empty answer sends the caller
    debugging the corpus instead of their permissions."""
    from rag.config import get_settings as _get

    settings = _get()
    original = settings.api_default_departments
    settings.api_default_departments = []
    try:
        response = await client.post(
            "/chat", json={"message": "What is the PTO policy?"}
        )
        assert response.status_code == 403
        assert "department scope" in response.json()["detail"]
    finally:
        settings.api_default_departments = original


@live
async def test_scope_is_enforced_on_the_answer(client, facts):
    response = await client.post(
        "/chat",
        json={"message": "How many days of paid sick leave do employees get?"},
        headers={HEADER: facts.other_department},
    )
    body = response.json()
    assert all(c["department"] == facts.other_department for c in body["citations"])
    assert all("HR/" not in c["doc_id"] for c in body["citations"])


@live
async def test_repeat_question_is_served_from_cache(client, facts):
    payload = {"message": "What is the annual learning and development limit?"}
    headers = {HEADER: "HR"}

    first = await client.post("/chat", json=payload, headers=headers)
    assert first.json()["cached"] is False

    second = await client.post("/chat", json=payload, headers=headers)
    body = second.json()
    assert body["cached"] is True
    assert body["answer"] == first.json()["answer"]


@live
async def test_cached_answer_does_not_cross_scopes(client, facts):
    """A cached HR answer must not surface for an IT-scoped caller."""
    payload = {"message": "What does the medical plan cost per month?"}
    await client.post("/chat", json=payload, headers={HEADER: "HR"})

    response = await client.post(
        "/chat", json=payload, headers={HEADER: facts.other_department}
    )
    body = response.json()
    assert body["cached"] is False
    assert all(c["department"] == facts.other_department for c in body["citations"])


@live
async def test_follow_up_uses_supplied_history(client, facts):
    response = await client.post(
        "/chat",
        json={
            "message": f"What about {facts.entry_tier}?",
            "history": [
                {"role": "user",
                 "content": f"What is the {facts.top_tier} tier price per seat?"},
                {"role": "assistant",
                 "content": f"{facts.top_tier} is ${facts.top_tier_price} per seat per month."},
            ],
        },
        headers={HEADER: "sales"},
    )
    body = response.json()
    assert body["answered"] is True
    assert facts.entry_tier_price in body["answer"]


@live
async def test_stats_reports_traffic(client, facts):
    await client.post(
        "/chat", json={"message": "What receipts are required for expenses?"},
        headers={HEADER: "finance"},
    )
    response = await client.get("/stats")
    body = response.json()
    assert body["queries"] >= 1
    assert body["latency_ms"]["p50"] > 0
    assert "hit_rate" in body["cache"]
    assert body["departments"] == get_settings().departments


@live
async def test_stream_emits_progress_then_the_answer(client, facts):
    async with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "How much annual leave do employees accrue?"},
        headers={HEADER: "HR"},
    ) as response:
        assert response.status_code == 200
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert "event: progress" in body
    assert "event: answer" in body
    assert body.rstrip().endswith("event: done\ndata: {}")
