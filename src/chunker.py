"""
Obsidian Markdown Document Chunker
Splits markdown documents into semantically coherent chunks while preserving
precise section headers, character offsets, and source metadata for exact passage highlighting.
"""

from dataclasses import dataclass
import re
from typing import List

try:
    from src.loader import NoteDocument
except ModuleNotFoundError:
    from loader import NoteDocument


@dataclass
class NoteChunk:
    """Represents a chunk of a Markdown note with rich source metadata."""
    chunk_id: str
    filename: str
    relative_path: str
    title: str
    heading: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int

    def to_metadata_dict(self) -> dict:
        """Serializes metadata for vector database storage (ChromaDB)."""
        return {
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "relative_path": self.relative_path,
            "title": self.title,
            "heading": self.heading,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


def split_text_into_sections(markdown_text: str) -> List[tuple[str, str, int, int]]:
    """
    Splits markdown content into structural sections based on markdown headers (#, ##, ###).
    
    Returns:
        List of tuples: (heading_title, section_text, start_char, end_char)
    """
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(markdown_text))
    
    if not matches:
        return [("General", markdown_text.strip(), 0, len(markdown_text))]

    sections = []
    
    # Text before the first header (if any)
    first_match = matches[0]
    if first_match.start() > 0:
        pre_text = markdown_text[:first_match.start()].strip()
        if pre_text:
            sections.append(("Introduction", pre_text, 0, first_match.start()))

    for i, match in enumerate(matches):
        heading_title = match.group(2).strip()
        start_idx = match.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        
        section_text = markdown_text[start_idx:end_idx].strip()
        if section_text:
            sections.append((heading_title, section_text, start_idx, end_idx))

    return sections


def chunk_document(
    doc: NoteDocument,
    target_chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[NoteChunk]:
    """
    Chunks a single NoteDocument into NoteChunk objects.
    Preserves heading context and exact character offsets in the original note.
    
    Args:
        doc: The NoteDocument to chunk.
        target_chunk_size: Preferred maximum character length of each chunk.
        chunk_overlap: Character overlap between consecutive split chunks.
        
    Returns:
        List[NoteChunk]: The resulting list of chunks.
    """
    sections = split_text_into_sections(doc.content)
    chunks: List[NoteChunk] = []
    global_chunk_idx = 0

    for heading, sec_text, sec_start, sec_end in sections:
        # If the entire section fits comfortably within the target size
        if len(sec_text) <= target_chunk_size + 150:
            pos = doc.content.find(sec_text, sec_start)
            start_pos = pos if pos != -1 else sec_start
            end_pos = start_pos + len(sec_text)

            chunk = NoteChunk(
                chunk_id=f"{doc.filename}#c{global_chunk_idx}",
                filename=doc.filename,
                relative_path=doc.relative_path,
                title=doc.title,
                heading=heading,
                chunk_index=global_chunk_idx,
                text=sec_text,
                start_char=start_pos,
                end_char=end_pos,
            )
            chunks.append(chunk)
            global_chunk_idx += 1
        else:
            # Split section by paragraphs (\n\n) or sliding window
            paragraphs = [p.strip() for p in sec_text.split("\n\n") if p.strip()]
            current_buffer = []
            current_len = 0

            for p in paragraphs:
                if current_len + len(p) > target_chunk_size and current_buffer:
                    sub_text = "\n\n".join(current_buffer)
                    pos = doc.content.find(sub_text, sec_start)
                    start_pos = pos if pos != -1 else sec_start
                    end_pos = start_pos + len(sub_text)

                    chunk = NoteChunk(
                        chunk_id=f"{doc.filename}#c{global_chunk_idx}",
                        filename=doc.filename,
                        relative_path=doc.relative_path,
                        title=doc.title,
                        heading=heading,
                        chunk_index=global_chunk_idx,
                        text=sub_text,
                        start_char=start_pos,
                        end_char=end_pos,
                    )
                    chunks.append(chunk)
                    global_chunk_idx += 1
                    current_buffer = [p]
                    current_len = len(p)
                else:
                    current_buffer.append(p)
                    current_len += len(p) + 2

            if current_buffer:
                sub_text = "\n\n".join(current_buffer)
                pos = doc.content.find(sub_text, sec_start)
                start_pos = pos if pos != -1 else sec_start
                end_pos = start_pos + len(sub_text)

                chunk = NoteChunk(
                    chunk_id=f"{doc.filename}#c{global_chunk_idx}",
                    filename=doc.filename,
                    relative_path=doc.relative_path,
                    title=doc.title,
                    heading=heading,
                    chunk_index=global_chunk_idx,
                    text=sub_text,
                    start_char=start_pos,
                    end_char=end_pos,
                )
                chunks.append(chunk)
                global_chunk_idx += 1

    return chunks


def chunk_vault(
    documents: List[NoteDocument],
    target_chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[NoteChunk]:
    """Chunks an entire list of NoteDocuments."""
    all_chunks: List[NoteChunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc, target_chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)
    return all_chunks


if __name__ == "__main__":
    from pathlib import Path
    try:
        from src.loader import load_vault
    except ModuleNotFoundError:
        from loader import load_vault

    docs = load_vault("data/demo_vault")
    chunks = chunk_vault(docs)
    print(f"Total notes: {len(docs)}, Total generated chunks: {len(chunks)}")
    print("\nSample chunks:")
    for c in chunks[:4]:
        print(f"[{c.chunk_id}] {c.filename} | Heading: '{c.heading}' | Chars: {c.start_char}-{c.end_char} ({len(c.text)} chars)")
        print(f"Preview: {c.text[:80]}...\n")
