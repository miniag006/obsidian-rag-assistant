"""
Automated unit & integration tests for Obsidian Vault RAG Assistant
"""

from pathlib import Path
import unittest

from src.loader import load_vault, extract_title_from_markdown, NoteDocument
from src.chunker import chunk_document, chunk_vault, split_text_into_sections, NoteChunk
from src.utils import highlight_passage_in_markdown, normalize_whitespace
from src.rag import format_context_for_llm, extract_supporting_sources_and_chunks, FALLBACK_MESSAGE, RAGResponse
from src.vectorstore import RetrievedChunk


class TestObsidianRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.demo_vault_dir = Path("data/demo_vault")

    def test_demo_vault_loader(self):
        """Verify all 5 demo vault markdown notes load properly."""
        docs = load_vault(self.demo_vault_dir)
        self.assertEqual(len(docs), 5)
        
        filenames = {d.filename for d in docs}
        expected = {"RAG.md", "LLMs.md", "Embeddings.md", "Vector_Databases.md", "AI_Agents.md"}
        self.assertEqual(filenames, expected)

        for doc in docs:
            self.assertGreater(doc.char_count, 1000)
            self.assertGreater(doc.line_count, 20)
            self.assertTrue(len(doc.title) > 0)

    def test_markdown_title_extraction(self):
        """Verify H1 header extraction and fallback."""
        md_text = "# Custom Note Title\n\nSome body text..."
        self.assertEqual(extract_title_from_markdown(md_text, "fallback"), "Custom Note Title")

        no_header = "Just raw text without header."
        self.assertEqual(extract_title_from_markdown(no_header, "fallback"), "fallback")

    def test_chunker_preserves_metadata(self):
        """Verify chunker outputs non-empty chunks with correct character offsets and headings."""
        docs = load_vault(self.demo_vault_dir)
        chunks = chunk_vault(docs)
        self.assertGreater(len(chunks), 20)

        for chunk in chunks:
            self.assertTrue(chunk.chunk_id.startswith(chunk.filename))
            self.assertTrue(len(chunk.text) > 0)
            self.assertGreaterEqual(chunk.end_char, chunk.start_char)
            self.assertTrue(len(chunk.heading) > 0)

    def test_passage_highlighting(self):
        """Verify exact passage highlighting in full markdown note."""
        docs = load_vault(self.demo_vault_dir)
        rag_doc = next(d for d in docs if d.filename == "RAG.md")
        chunks = chunk_document(rag_doc)
        
        for chunk in chunks[:3]:
            highlighted, found = highlight_passage_in_markdown(
                full_content=rag_doc.content,
                passage=chunk.text,
                heading=chunk.heading
            )
            self.assertTrue(found, f"Passage highlighting failed for chunk: {chunk.chunk_id}")
            self.assertIn("📌 Retrieved Passage", highlighted)

    def test_context_formatting(self):
        """Verify retrieved chunks are formatted with source attribution tags."""
        sample_chunk = RetrievedChunk(
            chunk_id="RAG.md#c1",
            filename="RAG.md",
            relative_path="RAG.md",
            title="Retrieval-Augmented Generation (RAG)",
            heading="Why RAG is Essential",
            chunk_index=1,
            text="Hallucination Reduction is key.",
            start_char=100,
            end_char=132,
            distance=0.15,
            similarity_score=0.85
        )
        context_str = format_context_for_llm([sample_chunk])
        self.assertIn("Source File: RAG.md", context_str)
        self.assertIn("Why RAG is Essential", context_str)
        self.assertIn("Hallucination Reduction is key.", context_str)

    def test_precise_source_attribution_single_file(self):
        """Verify that when 5 chunks from different files are retrieved, only the used note is attributed."""
        c1 = RetrievedChunk("LLMs.md#1", "LLMs.md", "LLMs.md", "LLMs", "Pipeline", 1, "Pre-training, SFT, RLHF.", 0, 30, 0.1, 0.9)
        c2 = RetrievedChunk("RAG.md#1", "RAG.md", "RAG.md", "RAG", "Intro", 1, "RAG overview.", 0, 20, 0.2, 0.8)
        c3 = RetrievedChunk("AI_Agents.md#1", "AI_Agents.md", "AI_Agents.md", "Agents", "Intro", 1, "Agent overview.", 0, 20, 0.3, 0.7)
        
        raw_llm_response = "The LLM training pipeline has 3 stages: Pre-training, SFT, and RLHF.\n\n[SOURCES_USED: 1]"
        clean_ans, sources, supp_chunks = extract_supporting_sources_and_chunks(raw_llm_response, [c1, c2, c3])
        
        self.assertEqual(sources, ["LLMs.md"])
        self.assertEqual(len(supp_chunks), 1)
        self.assertEqual(supp_chunks[0].filename, "LLMs.md")
        self.assertNotIn("[SOURCES_USED", clean_ans)

    def test_precise_source_attribution_multi_file(self):
        """Verify that multi-file questions attribute all supporting files."""
        c1 = RetrievedChunk("RAG.md#1", "RAG.md", "RAG.md", "RAG", "Intro", 1, "RAG retrieves external notes.", 0, 25, 0.1, 0.9)
        c2 = RetrievedChunk("LLMs.md#1", "LLMs.md", "LLMs.md", "LLMs", "Intro", 1, "LLMs can hallucinate.", 0, 20, 0.15, 0.85)
        c3 = RetrievedChunk("Embeddings.md#1", "Embeddings.md", "Embeddings.md", "Embeddings", "Intro", 1, "Embeddings are vectors.", 0, 30, 0.2, 0.8)
        
        raw_llm_response = "RAG helps LLMs by grounding their generation in verified external context.\n\n[SOURCES_USED: 1, 2]"
        clean_ans, sources, supp_chunks = extract_supporting_sources_and_chunks(raw_llm_response, [c1, c2, c3])
        
        self.assertEqual(sources, ["RAG.md", "LLMs.md"])
        self.assertEqual(len(supp_chunks), 2)
        self.assertNotIn("[SOURCES_USED", clean_ans)


if __name__ == "__main__":
    unittest.main()
