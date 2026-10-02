from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Running from backend/ locally reads the repo-root .env; Docker Compose passes env vars.
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    database_url: str
    app_url: str = "http://localhost:3000"
    secret_key: str
    session_ttl_days: int = 7
    cookie_secure: bool = False

    google_client_id: str
    google_client_secret: str

    upload_dir: str = "uploads"

    # All models run locally in Ollama. The embedding model is set in app/rag/embeddings.py.
    ollama_base_url: str = "http://localhost:11434"
    # Pull missing models on startup (app/ollama.py). Tests turn it off.
    ollama_pull_models: bool = True
    # Any Ollama model with tool calling works.
    chat_model: str = "qwen3:8b"
    # Context window per request, in tokens (Ollama silently cuts longer prompts).
    chat_context_tokens: int = 16384
    # Let thinking models reason before answering: slower, but better decisions.
    chat_think: bool = True
    # A local model can take minutes for one long reply.
    chat_timeout_seconds: float = 600

    # Calendar MCP server: the backend's only route to Google Calendar.
    calendar_mcp_url: str = "http://calendar-mcp:3333/mcp"
    calendar_mcp_timeout: float = 30.0

    @property
    def google_redirect_uri(self) -> str:
        # The frontend origin proxies /api/* to this service, so cookies stay same-origin.
        return f"{self.app_url}/api/auth/callback"


@lru_cache
def get_settings() -> Settings:
    return Settings()
