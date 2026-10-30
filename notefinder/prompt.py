import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    """The extraction prompt sent to Claude with each page image.

    :ivar text: The prompt text.
    """
    text: str

    @classmethod
    def from_file(cls, path: Path) -> "Prompt":
        """Read a prompt from a UTF-8 text file.

        :param path: The file to read.
        :returns: The prompt.
        """
        return cls(path.read_text(encoding="utf-8"))

    @property
    def version(self) -> str:
        """A short SHA-256 of the text, so any edit changes it.

        Stored with each page to tell which prompt extracted it.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]
