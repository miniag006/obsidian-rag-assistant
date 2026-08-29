"""
Obsidian Vault Markdown Loader
Loads and parses Markdown files (.md) from a given directory using pathlib.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class NoteDocument:
    """Represents a single loaded Markdown note from an Obsidian vault."""
    filename: str
    relative_path: str
    title: str
    content: str
    char_count: int
    line_count: int


def extract_title_from_markdown(content: str, fallback_title: str) -> str:
    """
    Extracts the first H1 header (# Title) from Markdown content,
    falling back to the note's filename if no H1 header is present.
    """
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and len(stripped) > 2:
            return stripped[2:].strip()
    return fallback_title


def load_vault(vault_dir: Path | str) -> List[NoteDocument]:
    """
    Recursively scans the given directory for .md files, reads their text,
    and returns a list of NoteDocument objects.
    
    Args:
        vault_dir: Path or string pointing to the Obsidian vault directory.
        
    Returns:
        List[NoteDocument]: Successfully loaded Markdown documents.
        
    Raises:
        FileNotFoundError: If the provided directory does not exist.
        ValueError: If no markdown files are found in the directory.
    """
    vault_path = Path(vault_dir).resolve()
    
    if not vault_path.exists():
        raise FileNotFoundError(f"Vault directory not found: {vault_path}")
    if not vault_path.is_dir():
        raise NotADirectoryError(f"Vault path is not a directory: {vault_path}")

    # Discover all .md files (case-insensitive for safety)
    md_files = sorted([
        p for p in vault_path.rglob("*")
        if p.is_file() and p.suffix.lower() == ".md" and not p.name.startswith(".")
    ])

    if not md_files:
        raise ValueError(f"No Markdown (.md) files found in vault: {vault_path}")

    documents: List[NoteDocument] = []

    for file_path in md_files:
        try:
            # Read with utf-8 encoding, replacing any invalid characters safely
            content = file_path.read_text(encoding="utf-8", errors="replace")
            
            # Skip completely empty files
            if not content.strip():
                continue

            rel_path = str(file_path.relative_to(vault_path))
            filename = file_path.name
            title = extract_title_from_markdown(content, fallback_title=file_path.stem)
            lines = content.splitlines()

            doc = NoteDocument(
                filename=filename,
                relative_path=rel_path,
                title=title,
                content=content,
                char_count=len(content),
                line_count=len(lines)
            )
            documents.append(doc)
        except Exception as e:
            print(f"Warning: Could not read file '{file_path}': {e}")
            continue

    if not documents:
        raise ValueError(f"Vault at {vault_path} contains only empty or unreadable Markdown files.")

    return documents


if __name__ == "__main__":
    demo_path = Path("data/demo_vault")
    notes = load_vault(demo_path)
    print(f"Successfully loaded {len(notes)} notes from {demo_path}:")
    for note in notes:
        print(f" - {note.filename} (Title: '{note.title}', {note.char_count} chars, {note.line_count} lines)")
