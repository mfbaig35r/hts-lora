"""Tests for the FastAPI serving wrapper."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hts_lora.serving.app import create_app
from hts_lora.serving.client import MLXClient, UpstreamResponse
from hts_lora.serving.prompts import THINK_PREFIX, build_messages, render_prompt

# ── Sample model outputs ────────────────────────────────────────────────────

GOOD_OUTPUT = (
    "<think>\n</think>\n\n"
    "Chapter 85: ELECTRICAL MACHINERY AND EQUIPMENT\n"
    "Heading 85.44: Insulated wire, cable\n"
    "Subheading 8544.42: Other electric conductors\n"
    "HTS Code: 8544.42.9000\n"
    "\n"
    "Reasoning: The product is an insulated electrical conductor of copper.\n"
    "\n"
    "Provides for: Insulated electric conductors fitted with connectors"
)

ABSTAIN_OUTPUT = (
    "<think>\n</think>\n\n"
    "Cannot classify: The product description is too vague to determine a "
    "specific HTS classification."
)

GARBAGE_OUTPUT = "<think>\n</think>\n\nThis is just random nonsense with no structure."


# ── Fakes ───────────────────────────────────────────────────────────────────

class FakeMLXClient(MLXClient):
    """In-memory MLXClient that returns canned responses without HTTP."""

    def __init__(
        self,
        canned: str = GOOD_OUTPUT,
        latency_ms: int = 1234,
        reachable: bool = True,
        raise_on_complete: bool = False,
    ) -> None:
        self.base_url = "http://fake/v1"
        self.model = "fake-model"
        self.timeout = 1.0
        self._canned = canned
        self._latency_ms = latency_ms
        self._reachable = reachable
        self._raise = raise_on_complete
        self.last_messages: list[dict[str, str]] | None = None
        self.last_prompt: str | None = None

    def complete(  # type: ignore[override]
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> UpstreamResponse:
        if self._raise:
            raise ConnectionError("upstream is down")
        self.last_messages = messages
        return UpstreamResponse(text=self._canned, latency_ms=self._latency_ms)

    def complete_prompt(  # type: ignore[override]
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> UpstreamResponse:
        if self._raise:
            raise ConnectionError("upstream is down")
        self.last_prompt = prompt
        return UpstreamResponse(text=self._canned, latency_ms=self._latency_ms)

    def is_reachable(self) -> bool:  # type: ignore[override]
        return self._reachable


# ── Prompt builder tests ────────────────────────────────────────────────────


class TestBuildMessages:
    def test_minimal_variant(self):
        messages = build_messages(description="copper wire")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "HTS" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Product: copper wire"

    def test_rich_variant_with_all_fields(self):
        messages = build_messages(
            description="copper wire",
            materials="copper, PVC",
            use="residential wiring",
            country_of_origin="Mexico",
        )
        user = messages[1]["content"]
        assert "Product: copper wire" in user
        assert "Materials: copper, PVC" in user
        assert "Use: residential wiring" in user
        assert "Country of origin: Mexico" in user

    def test_rich_variant_with_partial_fields(self):
        messages = build_messages(description="cotton shirt", materials="100% cotton")
        user = messages[1]["content"]
        assert "Product: cotton shirt" in user
        assert "Materials: 100% cotton" in user
        assert "Use:" not in user
        assert "Country of origin:" not in user

    def test_system_prompt_includes_thinking_off(self):
        messages = build_messages(description="anything")
        assert "detailed thinking off" in messages[0]["content"]


class TestRenderPrompt:
    def test_render_includes_special_tokens(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        prompt = render_prompt(messages)
        assert prompt.startswith("<|begin_of_text|>")
        assert "<|start_header_id|>system<|end_header_id|>\n\nsys<|eot_id|>" in prompt
        assert "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>" in prompt
        assert "<|start_header_id|>assistant<|end_header_id|>" in prompt

    def test_render_appends_think_prefix(self):
        prompt = render_prompt([{"role": "user", "content": "x"}])
        assert prompt.endswith(THINK_PREFIX)
        # think prefix must come AFTER the assistant header
        assistant_header_idx = prompt.find("<|start_header_id|>assistant<|end_header_id|>")
        think_idx = prompt.find(THINK_PREFIX)
        assert think_idx > assistant_header_idx

    def test_render_preserves_content_order(self):
        messages = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        prompt = render_prompt(messages)
        assert prompt.index("first") < prompt.index("second")


# ── /classify endpoint tests ────────────────────────────────────────────────


class TestClassifyEndpoint:
    def test_successful_classification(self):
        fake = FakeMLXClient(canned=GOOD_OUTPUT, latency_ms=1842)
        client = TestClient(create_app(client=fake))

        resp = client.post(
            "/classify",
            json={
                "description": "insulated copper wire",
                "materials": "copper, PVC",
                "use": "residential wiring",
                "country_of_origin": "Mexico",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["parse_ok"] is True
        assert body["is_abstention"] is False
        assert body["hts_code"] == "8544.42.9000"
        assert body["chapter"]["code"] == "85"
        assert body["heading"]["code"] == "85.44"
        assert body["subheading"]["code"] == "8544.42"
        assert "insulated electrical conductor" in body["reasoning"]
        assert body["raw"] is None  # only populated on parse failure
        assert body["latency_ms"] == 1842
        assert body["model"]

    def test_abstention(self):
        fake = FakeMLXClient(canned=ABSTAIN_OUTPUT)
        client = TestClient(create_app(client=fake))

        resp = client.post("/classify", json={"description": "thing"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_abstention"] is True
        assert body["parse_ok"] is True
        assert body["hts_code"] is None
        assert "too vague" in body["abstention_reason"]

    def test_parse_failure_returns_200_with_raw(self):
        fake = FakeMLXClient(canned=GARBAGE_OUTPUT)
        client = TestClient(create_app(client=fake))

        resp = client.post("/classify", json={"description": "wire"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["parse_ok"] is False
        assert body["hts_code"] is None
        assert body["raw"] is not None
        assert "random nonsense" in body["raw"]

    def test_upstream_unreachable_returns_503(self):
        fake = FakeMLXClient(raise_on_complete=True)
        client = TestClient(create_app(client=fake))

        resp = client.post("/classify", json={"description": "wire"})
        assert resp.status_code == 503
        assert "upstream" in resp.json()["detail"].lower()

    def test_classify_uses_minimal_variant_when_no_extras(self):
        fake = FakeMLXClient(canned=GOOD_OUTPUT)
        client = TestClient(create_app(client=fake))

        client.post("/classify", json={"description": "copper wire only"})
        assert fake.last_prompt is not None
        assert "Product: copper wire only" in fake.last_prompt
        assert "Materials:" not in fake.last_prompt
        # think prefix must be present so the model continues from it
        assert THINK_PREFIX in fake.last_prompt

    def test_classify_uses_rich_variant_when_extras_present(self):
        fake = FakeMLXClient(canned=GOOD_OUTPUT)
        client = TestClient(create_app(client=fake))

        client.post(
            "/classify",
            json={"description": "copper wire", "materials": "copper"},
        )
        assert fake.last_prompt is not None
        assert "Product: copper wire" in fake.last_prompt
        assert "Materials: copper" in fake.last_prompt

    def test_description_is_required(self):
        fake = FakeMLXClient()
        client = TestClient(create_app(client=fake))

        resp = client.post("/classify", json={})
        assert resp.status_code == 422  # FastAPI validation error


# ── /health endpoint tests ──────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_ok_when_upstream_reachable(self):
        fake = FakeMLXClient(reachable=True)
        client = TestClient(create_app(client=fake))

        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["upstream_reachable"] is True
        assert body["model"]

    def test_health_503_when_upstream_unreachable(self):
        fake = FakeMLXClient(reachable=False)
        client = TestClient(create_app(client=fake))

        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["upstream_reachable"] is False


# ── / index endpoint tests ──────────────────────────────────────────────────


class TestIndexEndpoint:
    def test_index_returns_html(self):
        fake = FakeMLXClient()
        client = TestClient(create_app(client=fake))

        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "HTS Classification API" in resp.text
        assert "/classify" in resp.text

    def test_openapi_docs_available(self):
        fake = FakeMLXClient()
        client = TestClient(create_app(client=fake))

        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        assert "/classify" in spec["paths"]
        assert "/health" in spec["paths"]
