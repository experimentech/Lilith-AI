import pytest

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


@pytest.fixture()
def adapter_with_stubs():
    store = FakeStore()
    session = FakeSession(store)
    weather = StubWeatherClient()
    news = StubNewsClient()
    adapter = MCPAdapter(
        session_factory=lambda client_id, context_id, config: session,
        weather_client=weather,
        news_client=news,
    )
    return adapter, store, session, weather, news


def test_teach_feedback_and_stats(adapter_with_stubs):
    adapter, store, session, weather, news = adapter_with_stubs

    # Teach routes to store
    pattern_id = adapter.handle_teach("alice", trigger="hello", response="hi", intent="greeting")
    assert pattern_id == "pattern-1"
    assert store.add_calls == [("hello", "hi", 0.5, "greeting")]

    # Feedback routes to session
    adapter.handle_upvote("alice", pattern_id, strength=0.7)
    adapter.handle_downvote("alice", pattern_id, strength=0.4)
    assert session.upvotes == [(pattern_id, 0.7)]
    assert session.downvotes == [(pattern_id, 0.4)]

    # Stats returns user_store stats
    stats = adapter.handle_stats("alice")
    assert stats == {"total_patterns": 0}

    # Chat uses session
    resp = adapter.handle_chat("alice", "ping")
    assert resp.text == "echo: ping"
    assert resp.pattern_id == "p0"


def test_transient_tools_do_not_persist(adapter_with_stubs):
    adapter, store, session, weather, news = adapter_with_stubs

    before_add_calls = len(store.add_calls)
    before_up = len(session.upvotes)
    before_down = len(session.downvotes)

    w = adapter.handle_weather("alice", "London")
    n = adapter.handle_news("alice", "python")

    assert isinstance(w, WeatherReport)
    assert "stub weather" in w.summary
    assert "stub headline" in n

    # Ensure no store writes or feedback occurred
    assert len(store.add_calls) == before_add_calls
    assert len(session.upvotes) == before_up
    assert len(session.downvotes) == before_down

    # Ensure stub clients were called once each
    assert weather.calls == 1
    assert news.calls == 1
