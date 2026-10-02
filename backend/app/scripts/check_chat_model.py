"""Check the local Ollama models: one completion, one tool call and one embedding.

python -m app.scripts.check_chat_model
"""

import asyncio
import time

from app import llm, ollama
from app.config import get_settings
from app.rag import embeddings

# Room for a thinking model (CHAT_THINK) to reason before it answers.
CHECK_MAX_TOKENS = 2048

PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Call this tool when asked to.",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    },
}


async def main() -> None:
    settings = get_settings()
    print(f"OLLAMA_BASE_URL  {settings.ollama_base_url}")
    print(f"CHAT_MODEL       {settings.chat_model} (context {settings.chat_context_tokens})")
    print(f"EMBEDDING_MODEL  {embeddings.EMBEDDING_MODEL}")
    await ollama.prepare_models([embeddings.EMBEDDING_MODEL, settings.chat_model])

    try:
        started = time.monotonic()
        reply = await llm.complete(
            [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=CHECK_MAX_TOKENS
        )
        print(f"1. plain reply   {reply.content!r} ({time.monotonic() - started:.1f}s)")

        started = time.monotonic()
        reply = await llm.complete(
            [{"role": "user", "content": "Call the ping tool with n=7."}],
            tools=[PING_TOOL],
            tool_choice="required",
            max_tokens=CHECK_MAX_TOKENS,
        )
        tool_seconds = time.monotonic() - started

        started = time.monotonic()
        [vector] = await embeddings.embed_texts(["Docker volumes persist data."])
        print(f"3. embedding     {len(vector)} numbers ({time.monotonic() - started:.1f}s)")
    except (llm.LlmError, embeddings.EmbeddingError) as exc:
        print(f"FAILED           {exc}")
        raise SystemExit(1) from exc

    if reply.tool_calls and reply.tool_calls[0].name == "ping":
        print(f"2. tool call     ping({reply.tool_calls[0].arguments}) ({tool_seconds:.1f}s)")
        print("OK               these models can run the agents")
    else:
        print(f"2. tool call     none; the model replied {reply.content!r}")
        print("WARNING          no tool call: pick a CHAT_MODEL with tool support")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
