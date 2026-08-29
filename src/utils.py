"""
Utility functions for Obsidian Vault RAG Assistant:
- Exact passage highlighting in full Markdown notes
- Secure ZIP vault extraction
- Note content resolution
"""

import html
import os
from pathlib import Path
import re
import shutil
import zipfile
from typing import Optional, Tuple


def normalize_whitespace(text: str) -> str:
    """Normalizes consecutive whitespace while preserving essential characters."""
    return re.sub(r"\s+", " ", text.strip())


def highlight_passage_in_markdown(
    full_content: str,
    passage: str,
    heading: Optional[str] = None
) -> Tuple[str, bool]:
    """
    Locates the exact retrieved passage within the full original Markdown note
    and wraps it in an HTML <mark> highlight block.
    
    Args:
        full_content: The entire raw Markdown note text.
        passage: The retrieved chunk text.
        heading: Optional heading context to aid in localization.
        
    Returns:
        Tuple[str, bool]: (highlighted_markdown_or_html, was_found)
    """
    clean_passage = passage.strip()
    if not clean_passage:
        return full_content, False

    # Highlight styling that looks sharp in both light and dark themes
    highlight_start = (
        '<div style="border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.15); '
        'padding: 10px 14px; margin: 12px 0; border-radius: 4px; font-weight: 500;">\n'
        '<span style="font-size: 0.8em; text-transform: uppercase; color: #d97706; font-weight: bold; '
        'display: block; margin-bottom: 4px;">📌 Retrieved Passage Used in Answer:</span>\n\n'
    )
    highlight_end = "\n\n</div>"

    # 1. Direct exact match
    if clean_passage in full_content:
        highlighted = full_content.replace(
            clean_passage,
            f"{highlight_start}{clean_passage}{highlight_end}",
            1
        )
        return highlighted, True

    # 2. Try paragraph-by-paragraph match if multi-paragraph chunk
    paragraphs = [p.strip() for p in clean_passage.split("\n\n") if len(p.strip()) > 30]
    if paragraphs and paragraphs[0] in full_content:
        # Find first paragraph and last paragraph span
        first_p = paragraphs[0]
        last_p = paragraphs[-1]
        start_idx = full_content.find(first_p)
        end_idx = full_content.find(last_p, start_idx)
        if start_idx != -1 and end_idx != -1:
            end_idx += len(last_p)
            matched_span = full_content[start_idx:end_idx]
            highlighted = (
                full_content[:start_idx]
                + f"{highlight_start}{matched_span}{highlight_end}"
                + full_content[end_idx:]
            )
            return highlighted, True

    # 3. Fuzzy whitespace-tolerant regex match
    escaped_tokens = [re.escape(tok) for tok in clean_passage.split()]
    if escaped_tokens:
        fuzzy_pattern = r"\s+".join(escaped_tokens)
        try:
            match = re.search(fuzzy_pattern, full_content)
            if match:
                matched_span = match.group(0)
                highlighted = (
                    full_content[:match.start()]
                    + f"{highlight_start}{matched_span}{highlight_end}"
                    + full_content[match.end():]
                )
                return highlighted, True
        except Exception:
            pass

    # Fallback: Prepend highlight notice if exact span could not be isolated
    fallback_annotated = (
        f"{highlight_start}**Note**: The following retrieved passage was cited:\n\n"
        f"{clean_passage}{highlight_end}\n\n---\n\n### Original Note Content:\n\n{full_content}"
    )
    return fallback_annotated, False


def extract_vault_zip(zip_file_bytes, target_dir: Path) -> Path:
    """
    Safely extracts an uploaded Obsidian vault ZIP archive into target_dir.
    Guards against zip-slip / directory traversal vulnerabilities.
    
    Returns:
        Path: Directory containing the extracted markdown files.
    """
    target_dir = Path(target_dir).resolve()
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file_bytes) as z:
        for member in z.infolist():
            # Prevent zip-slip
            target_path = (target_dir / member.filename).resolve()
            if not str(target_path).startswith(str(target_dir)):
                raise ValueError(f"Malicious zip entry detected: {member.filename}")
            
            # Skip OS junk like __MACOSX or .DS_Store
            if "__MACOSX" in member.filename or member.filename.startswith("."):
                continue
                
            z.extract(member, target_dir)

    # If the zip created a single root folder containing the files, descend into it
    extracted_items = [p for p in target_dir.iterdir() if not p.name.startswith(".")]
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        # Check if this subfolder contains markdown files
        if list(extracted_items[0].rglob("*.md")):
            return extracted_items[0]

    return target_dir


if __name__ == "__main__":
    sample_note = "# RAG\n\nRetrieval-Augmented Generation is great.\n\n## Details\nIt reduces hallucinations.\n"
    sample_chunk = "It reduces hallucinations."
    result, found = highlight_passage_in_markdown(sample_note, sample_chunk)
    print(f"Match found: {found}")
    print("Result preview:")
    print(result)
