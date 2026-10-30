"""Every setting of the app, in one place.

The other modules take these values as arguments rather than importing them;
the CLI passes them in.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    # Searching
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    search_k: int = 5

    # Sync
    max_tokens: int = 8192
    api_max_retries: int = 5
    workers: int = 4
    prompt_path: Path = Path(__file__).parent / "prompt.txt"
    claude_model: str = "claude-haiku-5-5"
    dpi: int = 150
    embed_model: str = "BAAI/bge-base-en-v1.5"

    # Database
    db_url: str = "postgresql://notefinder:notefinder@localhost:5432/notefinder"


APP_CONFIG = AppConfig()
