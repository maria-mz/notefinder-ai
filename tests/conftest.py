import subprocess
from pathlib import Path

import pytest

import db


TEST_DB_NAME = "notefinder_test"
TEST_URL = f"postgresql://notefinder:notefinder@localhost:5433/{TEST_DB_NAME}"
COMPOSE = ["docker", "compose", "-f",
           str(Path(__file__).parent.parent / "docker-compose.test.yml")]


@pytest.fixture(scope="session")
def test_db_url():
    subprocess.run([*COMPOSE, "up", "-d", "--force-recreate", "--wait"], check=True)
    yield TEST_URL
    subprocess.run([*COMPOSE, "down"], check=True)


@pytest.fixture
def conn(test_db_url):
    """A connection to the test DB with an empty pages table."""
    connection = db.connect(test_db_url)
    assert connection.info.dbname == TEST_DB_NAME
    connection.execute("TRUNCATE pages")
    yield connection
    connection.close()
