import anyio
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from lilith.mcp_adapter import MCPAdapter, WeatherReport
from lilith.session import SessionResponse


class StubWeatherClient:
    def __init__(self):
        self.calls = 0

    def get_weather(self, location: str) -> WeatherReport:
        self.calls += 1
        return WeatherReport(location=location, summary="stub weather", temperature_c=21.5)


class StubNewsClient:
    def __init__(self):
        self.calls = 0

    def get_news(self, topic: str) -> str:
        self.calls += 1
        return f"stub headline about {topic}"


class FakeUserStore:
    def __init__(self):
        self.stats = {"total_patterns": 0}

    def get_stats(self):
        return self.stats

    def reset_user_data(self, keep_backup=True, bootstrap=False):
        return "backup.db"


class FakeStore:
    def __init__(self):
        self.user_store = FakeUserStore()
        self.add_calls = []

    def add_pattern(self, trigger_context: str, response_text: str, success_score: float = 0.5, intent: str = "general") -> str:
        self.add_calls.append((trigger_context, response_text, success_score, intent))
        return f"pattern-{len(self.add_calls)}"


class FakeSession:
    def __init__(self, store: FakeStore):
        self.store = store
        self.upvotes = []
        self.downvotes = []

    def process_message(self, message: str) -> SessionResponse:
        return SessionResponse(text=f"echo: {message}", pattern_id="p0", confidence=0.9)

    def upvote(self, pattern_id: str, strength: float = 1.0):
        self.upvotes.append((pattern_id, strength))

    def downvote(self, pattern_id: str, strength: float = 1.0):
        self.downvotes.append((pattern_id, strength))


class ClientContext(BaseModel):
    client_id: str
    context_id: str | None = None


class ChatRequest(ClientContext):
    message: str


class TeachRequest(ClientContext):
    trigger: str
    response: str
    intent: str = "general"


class FeedbackRequest(ClientContext):
    pattern_id: str
    strength: float = 1.0


class WeatherRequest(ClientContext):
    location: str


class NewsRequest(ClientContext):
    topic: str


class ResetRequest(ClientContext):
    keep_backup: bool = True
    bootstrap: bool = False


def build_app(adapter: MCPAdapter) -> FastAPI:
    app = FastAPI(title="Lilith MCP Adapter Test", version="0.1-test")

    @app.post("/chat")
    async def chat(req: ChatRequest):
        resp = await anyio.to_thread.run_sync(adapter.handle_chat, req.client_id, req.message, req.context_id)
        return {
            "text": resp.text,
            "pattern_id": resp.pattern_id,
            "confidence": resp.confidence,
            "is_fallback": resp.is_fallback,
            "is_low_confidence": resp.is_low_confidence,
            "source": resp.source,
            "learned_fact": resp.learned_fact,
        }

    @app.post("/teach")
    async def teach(req: TeachRequest):
        pattern_id = await anyio.to_thread.run_sync(
            adapter.handle_teach,
            req.client_id,
            req.trigger,
            req.response,
            req.intent,
            req.context_id,
        )
        return {"pattern_id": pattern_id}

    @app.post("/feedback/upvote")
    async def upvote(req: FeedbackRequest):
        await anyio.to_thread.run_sync(adapter.handle_upvote, req.client_id, req.pattern_id, req.strength, req.context_id)
        return {"status": "ok"}

    @app.post("/feedback/downvote")
    async def downvote(req: FeedbackRequest):
        await anyio.to_thread.run_sync(adapter.handle_downvote, req.client_id, req.pattern_id, req.strength, req.context_id)
        return {"status": "ok"}

    @app.post("/stats")
    async def stats(req: ClientContext):
        return await anyio.to_thread.run_sync(adapter.handle_stats, req.client_id, req.context_id, True)

    @app.post("/weather")
    async def weather(req: WeatherRequest):
        report: WeatherReport = await anyio.to_thread.run_sync(adapter.handle_weather, req.client_id, req.location, req.context_id)
        return {
            "location": report.location,
            "summary": report.summary,
            "temperature_c": report.temperature_c,
        }

    @app.post("/news")
    async def news(req: NewsRequest):
        headline = await anyio.to_thread.run_sync(adapter.handle_news, req.client_id, req.topic, req.context_id)
        return {"headline": headline}

    @app.post("/reset")
    async def reset(req: ResetRequest):
        backup = await anyio.to_thread.run_sync(
            adapter.handle_reset_user,
            req.client_id,
            req.keep_backup,
            req.bootstrap,
            req.context_id,
        )
        if backup is None:
            raise HTTPException(status_code=400, detail="Reset not available for this caller")
        return {"backup_path": backup}

    return app


def test_mcp_server_endpoints_with_stub_session():
    store = FakeStore()
    session = FakeSession(store)
    weather = StubWeatherClient()
    news = StubNewsClient()
    adapter = MCPAdapter(
        session_factory=lambda client_id, context_id, config: session,
        weather_client=weather,
        news_client=news,
        max_sessions=4,
        session_ttl_seconds=60,
    )

    app = build_app(adapter)
    client = TestClient(app)

    # Chat should echo
    chat_resp = client.post("/chat", json={"client_id": "alice", "message": "ping"}).json()
    assert chat_resp["text"] == "echo: ping"
    assert chat_resp["pattern_id"] == "p0"

    # Teach should route to add_pattern via fallback
    teach_resp = client.post(
        "/teach",
        json={"client_id": "alice", "trigger": "hello", "response": "hi", "intent": "greeting"},
    ).json()
    assert teach_resp["pattern_id"] == "pattern-1"
    assert store.add_calls == [("hello", "hi", 0.5, "greeting")]

    # Feedback routes
    client.post("/feedback/upvote", json={"client_id": "alice", "pattern_id": "pattern-1", "strength": 0.7})
    client.post("/feedback/downvote", json={"client_id": "alice", "pattern_id": "pattern-1", "strength": 0.4})
    assert session.upvotes == [("pattern-1", 0.7)]
    assert session.downvotes == [("pattern-1", 0.4)]

    # Weather/news are transient
    before_add = len(store.add_calls)
    w = client.post("/weather", json={"client_id": "alice", "location": "London"}).json()
    n = client.post("/news", json={"client_id": "alice", "topic": "python"}).json()
    assert "stub weather" in w["summary"]
    assert "stub headline" in n["headline"]
    assert len(store.add_calls) == before_add
    assert weather.calls == 1
    assert news.calls == 1

    # Stats include user stats and cache stats
    stats = client.post("/stats", json={"client_id": "alice"}).json()
    assert stats.get("total_patterns") == 0
    assert "mcp_cache" in stats
    assert "hits" in stats["mcp_cache"]

    # Cache evicts on TTL/size; ensure cache bookkeeping is accessible
    cache_stats = adapter.get_cache_stats()
    assert cache_stats["max_sessions"] == 4.0
