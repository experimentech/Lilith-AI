from __future__ import annotations

from lilith.mcp_tool_stream import MCPToolInfo, build_generic_arguments, score_tool_for_text


def test_build_generic_arguments_text_required():
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    args = build_generic_arguments("hello world", schema)
    assert args == {"text": "hello world"}


def test_build_generic_arguments_location_required_extracts_in_clause():
    schema = {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    }
    args = build_generic_arguments("what's the weather in London today?", schema)
    assert args is not None
    assert args.get("location") == "London"


def test_build_generic_arguments_unsatisfied_required_returns_none():
    schema = {
        "type": "object",
        "properties": {"foo": {"type": "string"}},
        "required": ["foo"],
    }
    args = build_generic_arguments("hello", schema)
    assert args is None


def test_score_tool_for_text_uses_metadata_tokens_only():
    tool = MCPToolInfo(
        server="demo",
        name="news.get",
        description="Get current news for a topic",
        input_schema={},
    )
    s = score_tool_for_text(tool, "any news about spacex?")
    assert s > 0.0
