import os

from dotenv import load_dotenv

# Load variables from a local .env file if present.
load_dotenv()


class Settings:
    """Simple settings container for environment-based configuration."""

    openai_api_key: str
    model_name: str
    discord_webhook_url: str | None
    langsmith_project: str

    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

        # LangSmith / LangChain tracing configuration.
        default_project = "website-support-agent"
        self.langsmith_project = os.getenv("LANGCHAIN_PROJECT", default_project)

        # Ensure tracing v2 is enabled and project name is set so
        # LangGraph and LangChain runs are visible in LangSmith.
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", self.langsmith_project)


settings = Settings()
