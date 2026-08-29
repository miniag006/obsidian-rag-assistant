"""
Obsidian Vault RAG Knowledge Assistant - Streamlit Application
A student-level 1-week MVP for conversational RAG over Obsidian Markdown vaults.
Powered by Google Gemini API (Free Tier from Google AI Studio) & ChromaDB.
"""

import os
from pathlib import Path
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.loader import load_vault, NoteDocument
from src.chunker import chunk_vault, NoteChunk
from src.vectorstore import VaultVectorStore, RetrievedChunk, get_default_client
from src.rag import generate_rag_answer, RAGResponse, FALLBACK_MESSAGE
from src.utils import highlight_passage_in_markdown, extract_vault_zip

# Load environment variables from .env if present
load_dotenv()


def get_server_api_key() -> tuple[str, str]:
    """
    Safely retrieves API key and provider from Streamlit secrets or environment variables.
    Returns (api_key, provider_label)
    """
    # 1. Check Streamlit secrets
    try:
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
                return str(st.secrets["GEMINI_API_KEY"]).strip(), "Google Gemini"
            if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
                return str(st.secrets["OPENAI_API_KEY"]).strip(), "OpenAI"
    except Exception:
        pass

    # 2. Check environment variables
    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY", "").strip(), "Google Gemini"
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY", "").strip(), "OpenAI"

    return "", ""


# Streamlit Page Config
st.set_page_config(
    page_title="Obsidian RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished MVP look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "loaded_notes" not in st.session_state:
    st.session_state.loaded_notes = {}  # filename -> NoteDocument

if "vault_name" not in st.session_state:
    st.session_state.vault_name = None

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "viewing_source" not in st.session_state:
    st.session_state.viewing_source = None  # Dict of chunk details to display in viewer


def create_client_for_key(key: str) -> tuple[OpenAI, str, str]:
    """Returns configured OpenAI-compatible client, model name, and embedding model."""
    if key.startswith("sk-") and not os.getenv("GEMINI_API_KEY"):
        client = OpenAI(api_key=key)
        return client, "gpt-4o-mini", "text-embedding-3-small"
    else:
        # Default to Google Gemini API (Free from Google AI Studio)
        client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        return client, "gemini-1.5-flash", "text-embedding-004"


def initialize_vault(vault_path: Path, vault_label: str, active_api_key: str):
    """Loads markdown notes, chunks them, and builds ChromaDB vector embeddings."""
    if not active_api_key:
        st.sidebar.error("⚠️ An API Key is required to index the vault.")
        return

    with st.spinner(f"Ingesting and indexing '{vault_label}'..."):
        try:
            # 1. Load documents
            docs = load_vault(vault_path)
            notes_map = {doc.filename: doc for doc in docs}

            # 2. Chunk documents
            chunks = chunk_vault(docs)

            # 3. Create Client & Vector Store
            client, model_name, embedding_model = create_client_for_key(active_api_key)
            vector_store = VaultVectorStore(api_key=active_api_key, openai_client=client, embedding_model=embedding_model)
            indexed_count = vector_store.index_chunks(chunks)

            # 4. Save to session state
            st.session_state.loaded_notes = notes_map
            st.session_state.vector_store = vector_store
            st.session_state.vault_name = vault_label
            st.session_state.total_chunks = indexed_count
            st.session_state.active_model = model_name
            st.session_state.active_embedding = embedding_model
            st.session_state.viewing_source = None

            st.sidebar.success(f"Loaded {len(docs)} notes ({indexed_count} chunks)")
        except Exception as e:
            st.sidebar.error(f"Error loading vault: {str(e)}")


# Resolve API Key
server_key, provider = get_server_api_key()

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ Vault Settings")

    if server_key:
        st.success(f"🔒 **API Key:** Configured ({provider})")
        api_key = server_key
    else:
        st.markdown("🔑 **Google Gemini API Key** *(Free)*")
        user_key_input = st.text_input(
            "Enter Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Free key from Google AI Studio (https://aistudio.google.com/)"
        )
        st.caption("✨ [Get a Free Gemini API Key at aistudio.google.com](https://aistudio.google.com/) (no credit card needed).")
        api_key = user_key_input.strip()

    st.markdown("---")
    st.subheader("📚 Knowledge Vault")

    vault_source = st.radio(
        "Select Vault Source",
        options=["Preloaded Demo Vault", "Upload Custom Vault"],
        index=0
    )

    demo_vault_dir = Path("data/demo_vault")

    if vault_source == "Preloaded Demo Vault":
        st.caption("Preloaded notes: RAG, LLMs, Embeddings, Vector Databases, and AI Agents.")
        if st.button("🚀 Load Demo Vault", use_container_width=True):
            if not api_key:
                st.error("Please provide an API Key above to index the demo vault.")
            else:
                initialize_vault(demo_vault_dir, "Demo AI Knowledge Vault", api_key)

    elif vault_source == "Upload Custom Vault":
        st.caption("Upload a `.zip` archive containing your Obsidian `.md` notes, or individual `.md` files.")
        uploaded_file = st.file_uploader(
            "Upload Vault ZIP or Markdown",
            type=["zip", "md"],
            accept_multiple_files=False,
            help="Upload a zipped Obsidian vault folder or single .md note."
        )

        if uploaded_file and st.button("📥 Process & Index Vault", use_container_width=True):
            if not api_key:
                st.error("Please enter an API Key to index the vault.")
            else:
                temp_dir = Path("data/temp_uploaded_vault")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)

                if uploaded_file.name.endswith(".zip"):
                    extracted_path = extract_vault_zip(uploaded_file, temp_dir)
                    initialize_vault(extracted_path, f"Uploaded Vault ({uploaded_file.name})", api_key)
                elif uploaded_file.name.endswith(".md"):
                    note_path = temp_dir / uploaded_file.name
                    note_path.write_bytes(uploaded_file.getbuffer())
                    initialize_vault(temp_dir, f"Uploaded Note ({uploaded_file.name})", api_key)

    # Auto-load demo vault if API key is present and not yet loaded
    if api_key and st.session_state.vector_store is None:
        initialize_vault(demo_vault_dir, "Demo AI Knowledge Vault", api_key)

    # Vault Info Card
    st.markdown("---")
    st.subheader("📊 Vault Statistics")
    if st.session_state.vault_name:
        active_model = getattr(st.session_state, "active_model", "gemini-2.0-flash")
        active_emb = getattr(st.session_state, "active_embedding", "text-embedding-004")
        st.info(f"**Active Vault:** {st.session_state.vault_name}\n\n"
                f"- 📄 **Notes Indexed:** {len(st.session_state.loaded_notes)}\n"
                f"- 🧩 **Total Chunks:** {st.session_state.total_chunks}\n"
                f"- 🤖 **LLM Model:** `{active_model}`\n"
                f"- 🔍 **Embeddings:** `{active_emb}`")
        
        with st.expander("📑 View Indexed Notes"):
            for fname, note_doc in st.session_state.loaded_notes.items():
                st.markdown(f"- **{fname}** ({note_doc.char_count} chars)")
    else:
        st.warning("No vault loaded yet. Enter your API key and click 'Load Demo Vault'.")

    # Starter Questions
    st.markdown("---")
    st.subheader("💡 Example Questions")
    example_prompts = [
        "What is RAG and why is it useful?",
        "How does ChromaDB perform vector search?",
        "What are the main stages in an LLM training pipeline?",
        "What is the ReAct pattern for AI agents?",
        "What is the difference between Cosine Similarity and Dot Product?",
        "What is the recipe for baking chocolate cookies? (Out-of-vault test)"
    ]
    for prompt in example_prompts:
        if st.button(f"👉 {prompt}", key=f"ex_{prompt}", use_container_width=True):
            st.session_state.current_prompt = prompt

    # Reset Chat Button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.viewing_source = None
        st.rerun()


# --- Main Area ---
st.markdown('<div class="main-header">🧠 Obsidian Vault RAG Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Query your personal Obsidian knowledge base with verifiable source citations and exact passage highlighting. Powered by Google Gemini (Free).</div>', unsafe_allow_html=True)

# Document / Source Viewer Panel (if user clicked a source)
if st.session_state.viewing_source:
    src_info = st.session_state.viewing_source
    st.markdown("---")
    
    col_hdr, col_btn = st.columns([5, 1])
    with col_hdr:
        st.markdown(f"### 📖 Source Document: `{src_info['filename']}`")
        st.caption(f"**Section:** {src_info['heading']} | **Relevance Score:** {src_info['similarity_score']:.2%} | **Chunk ID:** `{src_info['chunk_id']}`")
    with col_btn:
        if st.button("✖️ Close Viewer", key="close_source_viewer", use_container_width=True):
            st.session_state.viewing_source = None
            st.rerun()

    # Locate note and highlight passage
    full_doc = st.session_state.loaded_notes.get(src_info["filename"])
    if full_doc:
        highlighted_content, found = highlight_passage_in_markdown(
            full_content=full_doc.content,
            passage=src_info["text"],
            heading=src_info["heading"]
        )
        st.markdown(highlighted_content, unsafe_allow_html=True)
    else:
        st.warning(f"Could not find full note for `{src_info['filename']}`.")
        st.markdown(f"**Retrieved Passage:**\n\n> {src_info['text']}")
    
    st.markdown("---")


# Display Chat Conversation
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Sources and Clickable Buttons for Assistant Responses
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            st.markdown("##### 📚 Sources Used:")
            
            retrieved_chunks = msg.get("retrieved_chunks", [])
            
            # Create a row of source buttons
            cols = st.columns(min(len(retrieved_chunks), 4) or 1)
            for c_idx, chunk in enumerate(retrieved_chunks):
                col = cols[c_idx % len(cols)]
                with col:
                    btn_label = f"📄 {chunk.filename} ({chunk.similarity_score:.0%})"
                    if st.button(
                        btn_label,
                        key=f"src_btn_{idx}_{c_idx}_{chunk.chunk_id}",
                        help=f"Click to inspect {chunk.filename} and highlight the retrieved passage under '{chunk.heading}'",
                        use_container_width=True
                    ):
                        st.session_state.viewing_source = {
                            "filename": chunk.filename,
                            "title": chunk.title,
                            "heading": chunk.heading,
                            "similarity_score": chunk.similarity_score,
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text
                        }
                        st.rerun()

            # Retrieval details expander
            with st.expander("🔍 View Retrieval & Context Details"):
                for i, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(
                        f"**[{i}] `{chunk.filename}`** — *Section:* `{chunk.heading}` | *Cosine Relevance:* `{chunk.similarity_score:.4f}`\n\n"
                        f"> {chunk.text}\n"
                    )


# User Question Input Handling
user_input = st.chat_input("Ask a question about your Obsidian notes...")

# Handle click from example questions
if "current_prompt" in st.session_state and st.session_state.current_prompt:
    user_input = st.session_state.current_prompt
    st.session_state.current_prompt = None

if user_input:
    # 1. Validation checks
    if not api_key:
        st.error("⚠️ Please provide a Gemini API Key in the sidebar to generate answers.")
    elif not st.session_state.vector_store:
        st.error("⚠️ No vault loaded. Please click 'Load Demo Vault' in the sidebar or upload a custom vault.")
    else:
        # 2. Append and render User message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 3. Retrieve chunks and generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching Obsidian vault and reasoning with Gemini..."):
                try:
                    # Semantic Search
                    retrieved = st.session_state.vector_store.search(user_input, top_k=4)
                    
                    # Generate Answer
                    client, model_name, _ = create_client_for_key(api_key)
                    rag_res: RAGResponse = generate_rag_answer(
                        query=user_input,
                        retrieved_chunks=retrieved,
                        openai_client=client,
                        model_name=model_name,
                        chat_history=st.session_state.messages
                    )

                    # Display Answer
                    st.markdown(rag_res.answer)

                    # Display Sources
                    if rag_res.sources and rag_res.retrieved_chunks:
                        st.markdown("##### 📚 Sources Used:")
                        cols = st.columns(min(len(rag_res.retrieved_chunks), 4) or 1)
                        for c_idx, chunk in enumerate(rag_res.retrieved_chunks):
                            col = cols[c_idx % len(cols)]
                            with col:
                                btn_label = f"📄 {chunk.filename} ({chunk.similarity_score:.0%})"
                                if st.button(
                                    btn_label,
                                    key=f"live_src_{c_idx}_{chunk.chunk_id}",
                                    help=f"Click to inspect {chunk.filename} and highlight the retrieved passage under '{chunk.heading}'",
                                    use_container_width=True
                                ):
                                    st.session_state.viewing_source = {
                                        "filename": chunk.filename,
                                        "title": chunk.title,
                                        "heading": chunk.heading,
                                        "similarity_score": chunk.similarity_score,
                                        "chunk_id": chunk.chunk_id,
                                        "text": chunk.text
                                    }
                                    st.rerun()

                    # Save Assistant message to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": rag_res.answer,
                        "sources": rag_res.sources,
                        "retrieved_chunks": rag_res.retrieved_chunks
                    })

                except Exception as e:
                    err_msg = f"❌ An error occurred: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
