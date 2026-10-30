import re

from notefinder.prompt import Prompt


def test_from_file_reads_the_text(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("Transcribe this page. ∀x", encoding="utf-8")

    assert Prompt.from_file(path) == Prompt("Transcribe this page. ∀x")


def test_version_is_a_short_hex_hash():
    assert re.fullmatch(r"[0-9a-f]{12}", Prompt("Transcribe this page.").version)


def test_version_changes_only_when_the_text_does():
    version = Prompt("Transcribe this page.").version

    assert Prompt("Transcribe this page.").version == version
    assert Prompt("Transcribe this page carefully.").version != version
