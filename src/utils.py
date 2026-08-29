"""
Utility functions for Obsidian Vault RAG Assistant:
- Multi-passage highlighting in full Markdown notes
- Unicode string sanitization
- Secure ZIP vault extraction
"""

from pathlib import Path
import re
import shutil
from typing import Any, List, Optional, Tuple
import unicodedata
import zipfile


def clean_unicode(text: str) -> str:
    """
    Sanitizes Unicode typographic smart quotes, dashes, and non-ASCII artifacts
    to prevent ASCII codec encoding errors across HTTP clients and LLM APIs.
    """
    if not text:
        return ""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u200b": "",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return unicodedata.normalize("NFKD", text)


def normalize_whitespace(text: str) -> str:
    """Normalizes consecutive whitespace while preserving essential characters."""
    return re.sub(r"\s+", " ", text.strip())


def highlight_passages_in_markdown(
    full_content: str,
    passages: Optional[List[str]] = None,
    offsets: Optional[List[Tuple[int, int]]] = None,
    chunks: Optional[List[Any]] = None,
) -> Tuple[str, int]:
    """
    Locates all retrieved passages within the full original Markdown note
    using exact character offsets and robust multi-strategy fallback matching.
    Applies HTML highlight blocks in reverse order to preserve exact character positions.
    """
    if not full_content:
        return full_content, 0

    chunk_items = []
    if chunks:
        for c in chunks:
            text = getattr(c, "text", "") or ""
            start = getattr(c, "start_char", -1)
            end = getattr(c, "end_char", -1)
            heading = getattr(c, "heading", "")
            chunk_items.append({"text": text, "start_char": start, "end_char": end, "heading": heading})
    elif passages:
        offsets_list = offsets or [(-1, -1)] * len(passages)
        for p, off in zip(passages, offsets_list):
            chunk_items.append({"text": p, "start_char": off[0], "end_char": off[1], "heading": ""})

    if not chunk_items:
        return full_content, 0

    spans = []  # List of (start_pos, end_pos)
    used_spans = set()

    for item in chunk_items:
        clean_text = item["text"].strip()
        if not clean_text:
            continue

        start_c = item.get("start_char", -1)
        end_c = item.get("end_char", -1)
        matched = False

        # 1. Exact character offset slice match
        if 0 <= start_c < end_c <= len(full_content):
            doc_slice = full_content[start_c:end_c]
            if doc_slice.strip() == clean_text or clean_text in doc_slice:
                span_key = (start_c, end_c)
                if span_key not in used_spans:
                    spans.append(span_key)
                    used_spans.add(span_key)
                    matched = True

        # 2. Exact substring search in full content
        if not matched:
            pos = full_content.find(clean_text)
            if pos != -1:
                span_key = (pos, pos + len(clean_text))
                if span_key not in used_spans:
                    spans.append(span_key)
                    used_spans.add(span_key)
                    matched = True

        # 3. Paragraph boundaries match (first paragraph and last paragraph)
        if not matched:
            paragraphs = [p.strip() for p in clean_text.split("\n\n") if len(p.strip()) > 20]
            if paragraphs:
                p_first = paragraphs[0]
                p_last = paragraphs[-1]
                s_idx = full_content.find(p_first)
                e_idx = full_content.find(p_last, s_idx) if s_idx != -1 else -1
                if s_idx != -1 and e_idx != -1:
                    e_idx += len(p_last)
                    span_key = (s_idx, e_idx)
                    if span_key not in used_spans:
                        spans.append(span_key)
                        used_spans.add(span_key)
                        matched = True

        # 4. Token-spaced regex match
        if not matched:
            tokens = [re.escape(tok) for tok in clean_text.split() if tok]
            if tokens:
                try:
                    pattern = r"\s+".join(tokens)
                    m = re.search(pattern, full_content)
                    if m:
                        span_key = (m.start(), m.end())
                        if span_key not in used_spans:
                            spans.append(span_key)
                            used_spans.add(span_key)
                            matched = True
                except Exception:
                    pass

    if not spans:
        return full_content, 0

    # Sort spans in ascending reading order to assign 1..N numbers
    sorted_spans = sorted(spans, key=lambda x: x[0])
    total_spans = len(sorted_spans)
    span_number_map = { span: idx + 1 for idx, span in enumerate(sorted_spans) }

    # Sort in descending order to apply in-place string replacements safely
    reverse_spans = sorted(spans, key=lambda x: x[0], reverse=True)
    content = full_content

    for s_pos, e_pos in reverse_spans:
        idx_num = span_number_map.get((s_pos, e_pos), 1)
        anchor_attr = ' id="first-retrieved-passage"' if idx_num == 1 else ""
        banner = (
            f'<div{anchor_attr} style="border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.15); '
            f'padding: 10px 14px; margin: 12px 0; border-radius: 4px; font-weight: 500;">\n'
            f'<span style="font-size: 0.8em; text-transform: uppercase; color: #d97706; font-weight: bold; '
            f'display: block; margin-bottom: 4px;">📌 Retrieved Passage [{idx_num}/{total_spans}]:</span>\n\n'
        )
        highlight_end = "\n\n</div>"
        
        content = content[:s_pos] + banner + content[s_pos:e_pos] + highlight_end + content[e_pos:]

    return content, total_spans


def highlight_passage_in_markdown(full_content: str, passage: str, heading: Optional[str] = None) -> Tuple[str, bool]:
    res, count = highlight_passages_in_markdown(full_content, passages=[passage])
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
