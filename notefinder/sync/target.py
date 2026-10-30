from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    """A PDF or folder to sync, and the notes root its pages are stored relative to.

    Build one with Target.resolve, which checks it.
    """
    root: Path
    path: Path

    @classmethod
    def resolve(cls, root: Path, path: Path | None = None) -> "Target":
        """The target at path (default: all of root), with both paths made absolute."""
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
        """The target as stored pages name it, e.g. 'os' or 'os/lecture3.pdf'; '' is all of root."""
        return "" if self.path == self.root else self.source_file(self.path)

    def pdfs(self) -> list[Path]:
        """Every PDF the target covers, in sorted order."""
        if self.path.is_file():
            return [self.path]
        return sorted(self.path.rglob("*.pdf"))

    def source_file(self, pdf: Path) -> str:
        """The value stored in pages.source_file for pdf, e.g. 'os/lecture3.pdf'."""
        return pdf.relative_to(self.root).as_posix()
