"""What a sync covers: a PDF or folder inside the notes root."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    """A PDF or folder to sync, and the notes root its pages are stored relative to.

    Build one with :meth:`resolve`, which checks it.

    :ivar root: The notes root. Absolute.
    :ivar path: The PDF or folder to sync. Absolute, and inside ``root``.
    """
    root: Path
    path: Path

    @classmethod
    def resolve(cls, root: Path, path: Path | None = None) -> "Target":
        """Check a target and make its paths absolute.

        :param root: The notes root.
        :param path: The PDF or folder to sync. Defaults to all of ``root``.
        :returns: The checked target.
        :raises FileNotFoundError: If ``root`` isn't a folder or ``path`` doesn't exist.
        :raises ValueError: If ``path`` is outside ``root``, or is a file but not a PDF.
        """
        root = root.resolve()
        path = root if path is None else path.resolve()

        if not root.is_dir():
            raise FileNotFoundError(f"notes folder not found: {root}")
        if not path.exists():
            raise FileNotFoundError(f"target not found: {path}")
        if not path.is_relative_to(root):
            raise ValueError(f"{path} is outside {root}")
        if path.is_file() and path.suffix != ".pdf":
            raise ValueError(f"{path} is not a PDF")

        return cls(root, path)

    @property
    def scope(self) -> str:
        """The target's path relative to ``root``, for matching stored pages.

        A stored page is in scope if its ``source_file`` equals this (a PDF) or is
        under it (a folder), e.g. ``"os"`` or ``"os/lecture3.pdf"``.
        ``""`` means all of ``root``.
        """
        if self.path == self.root:
            return ""
        return self.path.relative_to(self.root).as_posix()

    def pdfs(self) -> list[Path]:
        """List the PDFs the target covers.

        :returns: ``path`` itself if it's a file, else every PDF under it, sorted.
        """
        if self.path.is_file():
            return [self.path]
        return sorted(self.path.rglob("*.pdf"))

    def source_file(self, pdf: Path) -> str:
        """Name a PDF as ``pages.source_file`` stores it.

        :param pdf: An absolute path to a PDF inside ``root``.
        :returns: The path relative to ``root``, with ``/`` separators, e.g. ``"os/lecture3.pdf"``.
        :raises ValueError: If ``pdf`` isn't inside ``root``.
        """
        return pdf.relative_to(self.root).as_posix()
