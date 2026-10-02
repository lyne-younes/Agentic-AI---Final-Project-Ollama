"""Chat model access through Ollama.

Callers use the OpenAI message/tool format; this module converts to and from Ollama's /api/chat.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from app import ollama
from app.config import get_settings

# Must leave room for a long answer, which can arrive inside one tool call.
DEFAULT_MAX_TOKENS = 8192
_CUT_OFF_REASONS = {"length"}
# Ollama can't force a tool call, so `tool_choice="required"` is enforced by re-asking.
FORCED_TOOL_RETRIES = 1
FORCE_TOOL_PROMPT = (
    "(You must reply by calling one of the provided tools ({names}), not with plain text. "
    "Call the tool now.)"
)


class LlmError(Exception):
    """The chat model couldn't be reached or rejected the request."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LlmReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # True when generation stopped at max_tokens: the text or tool-call arguments are incomplete.
    truncated: bool = False


async def complete(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> LlmReply:
    """One chat completion. With `tool_choice="required"`, a plain-text reply is retried once."""
    reply = await _chat(messages, tools, max_tokens)
    if tools and tool_choice == "required":
        for _ in range(FORCED_TOOL_RETRIES):
            if reply.tool_calls or reply.truncated:
                break
            names = ", ".join(tool.get("function", {}).get("name", "?") for tool in tools)
            retry = [*messages]
            if reply.content:
                retry.append({"role": "assistant", "content": reply.content})
            retry.append({"role": "user", "content": FORCE_TOOL_PROMPT.format(names=names)})
            reply = await _chat(retry, tools, max_tokens)
    return reply


async def _chat(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int
) -> LlmReply:
    settings = get_settings()
    model = settings.chat_model
    if ollama.is_downloading(model):
        raise LlmError(f"The chat model {model} is still downloading")

    payload: dict[str, Any] = {
        "model": model,
        "messages": _to_ollama_messages(messages),
        "stream": False,
        # Reasoning comes back separately; agents only read the answer.
        "think": settings.chat_think,
        "options": {
            # Ollama's default context is too small for agent prompts.
            "num_ctx": settings.chat_context_tokens,
            "num_predict": max_tokens,
        },
    }
    if tools:
        payload["tools"] = tools

    try:
        data = await ollama.post("/api/chat", payload, timeout=settings.chat_timeout_seconds)
    except ollama.OllamaError as exc:
        raise LlmError(str(exc)) from exc

    message = data.get("message") or {}
    return LlmReply(
        content=message.get("content") or None,
        tool_calls=[
            ToolCall(
                id=call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                name=call.get("function", {}).get("name", ""),
                arguments=_parse_arguments(call.get("function", {}).get("arguments")),
            )
            for call in message.get("tool_calls") or []
        ],
        truncated=data.get("done_reason") in _CUT_OFF_REASONS,
    )


def _to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converts OpenAI-format messages to Ollama's format."""
    tool_names: dict[str, str] = {}
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call["function"]
                tool_names[call.get("id", "")] = function["name"]
                calls.append(
                    {
                        "function": {
                            "name": function["name"],
                            "arguments": _parse_arguments(function.get("arguments")),
                        }
                    }
                )
            converted.append({"role": "assistant", "content": content, "tool_calls": calls})
        elif role == "tool":
            tool_name = tool_names.get(message.get("tool_call_id", ""), "")
            converted.append({"role": "tool", "content": content, "tool_name": tool_name})
        else:
            converted.append({"role": role, "content": content})
    return converted


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, dict) else {}
