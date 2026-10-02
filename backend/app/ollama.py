"""Requests to the local Ollama server, and pulling missing models on startup."""

import json
import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Models being pulled right now; calls to them fail fast with a clear message.
_downloading: set[str] = set()


class OllamaError(Exception):
    """Ollama couldn't be reached, or it rejected the request."""


def is_downloading(model: str) -> bool:
    return _full_name(model) in _downloading


def any_downloading() -> bool:
    return bool(_downloading)


async def post(path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """One non-streaming request to Ollama's native API, e.g. `/api/chat` or `/api/embed`."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(_url(path), json=payload)
        if response.is_error:
            raise OllamaError(f"Ollama {path} returned {response.status_code}: {_error(response)}")
        data = response.json()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Ollama at {_base_url()} failed: {exc!r}") from exc
    except ValueError as exc:  # the body wasn't JSON
        raise OllamaError(f"Ollama {path} returned an unreadable reply: {exc}") from exc
    if not isinstance(data, dict):
        raise OllamaError(f"Ollama {path} returned an unexpected reply")
    return data


async def prepare_models(models: list[str]) -> None:
    """Startup job: pulls any of `models` Ollama doesn't have yet. Logs failures, never raises."""
    wanted = list(dict.fromkeys(_full_name(model) for model in models))
    _downloading.update(wanted)  # until we know which are installed; the check takes a moment
    try:
        installed = await _installed_models()
    except OllamaError as exc:
        _downloading.difference_update(wanted)
        logger.error("Can't reach Ollama at %s to check models: %s", _base_url(), exc)
        return

    missing = [model for model in wanted if model not in installed]
    _downloading.difference_update(set(wanted) - set(missing))
    for model in missing:
        try:
            await _pull(model)
        except OllamaError as exc:
            logger.error("Pulling %s failed: %s", model, exc)
        finally:
            _downloading.discard(model)
    if not missing:
        logger.info("Ollama models ready: %s", ", ".join(wanted))


async def _installed_models() -> set[str]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(_url("/api/tags"))
        response.raise_for_status()
        entries = response.json().get("models") or []
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise OllamaError(repr(exc)) from exc
    return {
        _full_name(name)
        for entry in entries
        for name in (entry.get("name"), entry.get("model"))
        if name
    }


async def _pull(model: str) -> None:
    logger.info("Pulling Ollama model %s (first start only; this can take several minutes)", model)
    layer, last_logged = None, -1
    try:
        # No read timeout: Ollama can go quiet for a while between progress lines (verifying).
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=None)) as client:
            async with client.stream("POST", _url("/api/pull"), json={"model": model}) as response:
                if response.is_error:
                    await response.aread()
                    raise OllamaError(f"{response.status_code}: {_error(response)}")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise OllamaError(event["error"])
                    total, completed = event.get("total"), event.get("completed")
                    if not total or completed is None:
                        continue
                    if event.get("digest") != layer:  # progress restarts for every layer
                        layer, last_logged = event.get("digest"), -1
                    percent = completed * 100 // total
                    if last_logged < 0 or percent >= last_logged + 10:
                        last_logged = percent
                        logger.info("Pulling %s: %s%% of %s MB", model, percent, total // 1_000_000)
    except httpx.HTTPError as exc:
        raise OllamaError(repr(exc)) from exc
    except ValueError as exc:
        raise OllamaError(f"unreadable progress from Ollama: {exc}") from exc
    logger.info("Pulled Ollama model %s", model)


def _full_name(model: str) -> str:
    """`llama3.1` and `llama3.1:latest` are the same model; Ollama lists the tagged form."""
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def _base_url() -> str:
    return get_settings().ollama_base_url.rstrip("/")


def _url(path: str) -> str:
    return f"{_base_url()}{path}"


def _error(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error") or response.text)
    except (ValueError, AttributeError):
        return response.text
