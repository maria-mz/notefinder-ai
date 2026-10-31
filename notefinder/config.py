"""App settings."""
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
#: The repo root. The eval paths assume the app runs from a checkout of the repo.
PROJECT_DIR = PACKAGE_DIR.parent


@dataclass(frozen=True)
class AppConfig:
    """App settings.

    :ivar query_prefix: Prefix used to embed queries.
    :ivar embed_model: The embedding model. On changed, sync re-embeds.

    :ivar search_k: How many results ``search`` shows.

    :ivar claude_model: The Claude model that reads pages. On changed, sync re-extracts.
    :ivar prompt_path: The extraction prompt. On changed, sync re-extracts.
    :ivar dpi: The resolution pages are rendered at. On changed, sync re-extracts.
    :ivar max_tokens: The most tokens Claude may write per page.
    :ivar api_max_retries: How many times a failed Claude request is retried.
    :ivar workers: How many pages a sync processes at once.
    :ivar eval_k: The k in recall@k.

    :ivar labels_path: The hand-labelled eval queries.
    :ivar runs_dir: Where each eval run's results are saved.

    :ivar db_url: The database. The default matches the local Docker Compose setup.
    """
    # embedding
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    embed_model: str = "BAAI/bge-base-en-v1.5"

    # searching
    search_k: int = 5

    # sync
    claude_model: str = "claude-sonnet-5-5"
    prompt_path: Path = PACKAGE_DIR / "prompt.txt"
    dpi: int = 150
    max_tokens: int = 8192
    api_max_retries: int = 5
    workers: int = 4

    # evaluating
    eval_k: int = 5
    labels_path: Path = PROJECT_DIR / "eval" / "labels.yaml"
    runs_dir: Path = PROJECT_DIR / "eval" / "runs"

    # database
    db_url: str = "postgresql://notefinder:notefinder@localhost:5432/notefinder"


APP_CONFIG = AppConfig()
