"""
Utility functions for Obsidian Vault RAG Assistant:
- Multi-passage highlighting in full Markdown notes
- Secure ZIP vault extraction
"""

from pathlib import Path
import re
import shutil
from typing import List, Optional, Tuple
import zipfile


def normalize_whitespace(text: str) -> str:
    """Normalizes consecutive whitespace while preserving essential characters."""
    return re.sub(r"\s+", " ", text.strip())


def highlight_passages_in_markdown(
    full_content: str,
    passages: List[str],
) -> Tuple[str, int]:
    """
    Locates all retrieved passages within the full original Markdown note
    and wraps each in an HTML <mark> highlight block.
    
    Args:
        full_content: The entire raw Markdown note text.
        passages: List of retrieved chunk text strings belonging to this note.
        
    Returns:
        Tuple[str, int]: (highlighted_markdown, number_of_passages_highlighted)
    """
    content = full_content
    found_count = 0
    total = len(passages)

    highlight_start_template = (
        '<div style="border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.15); '
        'padding: 10px 14px; margin: 12px 0; border-radius: 4px; font-weight: 500;">\n'
        '<span style="font-size: 0.8em; text-transform: uppercase; color: #d97706; font-weight: bold; '
        'display: block; margin-bottom: 4px;">📌 Retrieved Passage [{idx}/{total}]:</span>\n\n'
    )
    highlight_end = "\n\n</div>"

    for i, passage in enumerate(passages, start=1):
        clean_passage = passage.strip()
        if not clean_passage:
            continue

        banner = highlight_start_template.format(idx=i, total=total)

        # 1. Direct exact match
        if clean_passage in content:
            content = content.replace(clean_passage, f"{banner}{clean_passage}{highlight_end}", 1)
            found_count += 1
            continue

        # 2. Paragraph match
        paragraphs = [p.strip() for p in clean_passage.split("\n\n") if len(p.strip()) > 30]
        if paragraphs and paragraphs[0] in content and paragraphs[-1] in content:
            first_p = paragraphs[0]
            last_p = paragraphs[-1]
            start_idx = content.find(first_p)
            end_idx = content.find(last_p, start_idx)
            if start_idx != -1 and end_idx != -1:
                end_idx += len(last_p)
                matched_span = content[start_idx:end_idx]
                content = content[:start_idx] + f"{banner}{matched_span}{highlight_end}" + content[end_idx:]
                found_count += 1
                continue

        # 3. Fuzzy whitespace regex match
        escaped_tokens = [re.escape(tok) for tok in clean_passage.split()]
        if escaped_tokens:
            fuzzy_pattern = r"\s+".join(escaped_tokens)
            try:
                match = re.search(fuzzy_pattern, content)
                if match:
                    matched_span = match.group(0)
                    content = content[:match.start()] + f"{banner}{matched_span}{highlight_end}" + content[match.end():]
                    found_count += 1
            except Exception:
                pass

    return content, found_count


# Backwards compatibility alias
def highlight_passage_in_markdown(full_content: str, passage: str, heading: Optional[str] = None) -> Tuple[str, bool]:
    res, count = highlight_passages_in_markdown(full_content, [passage])
    return res, count > 0


def extract_vault_zip(zip_file_bytes, target_dir: Path) -> Path:
    """
    Safely extracts an uploaded Obsidian vault ZIP archive into target_dir.
    Guards against zip-slip / directory traversal vulnerabilities.
    """
    target_dir = Path(target_dir).resolve()
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file_bytes) as z:
        for member in z.infolist():
            target_path = (target_dir / member.filename).resolve()
            if not str(target_path).startswith(str(target_dir)):
                raise ValueError(f"Malicious zip entry detected: {member.filename}")
            
            if "__MACOSX" in member.filename or member.filename.startswith("."):
                continue
                
            z.extract(member, target_dir)

    extracted_items = [p for p in target_dir.iterdir() if not p.name.startswith(".")]
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        if list(extracted_items[0].rglob("*.md")):
            return extracted_items[0]

    return target_dir
