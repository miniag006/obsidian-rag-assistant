"""
Obsidian Vault RAG Knowledge Assistant - Streamlit Application
A student-level 1-week MVP for conversational RAG over Obsidian Markdown vaults.
Powered by Google Gemini API (Free Tier from Google AI Studio) & ChromaDB.
"""

import os
from pathlib import Path
import shutil
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.loader import load_vault, NoteDocument
from src.chunker import chunk_vault, NoteChunk
from src.vectorstore import VaultVectorStore, RetrievedChunk, get_default_client
from src.rag import generate_rag_answer, RAGResponse, FALLBACK_MESSAGE, contextualize_query_for_search
from src.utils import highlight_passages_in_markdown, extract_vault_zip

# Load environment variables from .env if present
load_dotenv()


def get_server_api_key() -> tuple[str, str]:
    """
    Safely retrieves API key and provider from Streamlit secrets or environment variables.
    Returns (api_key, provider_label)
    """
    try:
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
                return str(st.secrets["GEMINI_API_KEY"]).strip(), "Google Gemini"
            if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
                return str(st.secrets["OPENAI_API_KEY"]).strip(), "OpenAI"
    except Exception:
        pass

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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
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

if "active_viewer_msg_idx" not in st.session_state:
    st.session_state.active_viewer_msg_idx = None  # Message index currently displaying its viewer


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
        return client, "gemini-3.6-flash", "text-embedding-004"


def initialize_vault(vault_path: Path, vault_label: str, active_api_key: str):
    """Loads markdown notes, chunks them, and builds ChromaDB vector embeddings."""
    if not active_api_key:
        st.session_state.vault_load_error = "⚠️ An API Key is required to index the vault."
        st.session_state.vault_load_success = None
        return

    st.session_state.vault_load_error = None
    st.session_state.vault_load_success = None

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
            st.session_state.active_viewer_msg_idx = None
            st.session_state.vault_load_success = f"Loaded {len(docs)} notes ({indexed_count} chunks)"
            st.session_state.vault_load_error = None
        except Exception as e:
            st.session_state.vault_load_error = f"Error loading vault: {str(e)}"
            st.session_state.vault_load_success = None


# Resolve API Key
server_key, provider = get_server_api_key()

if "saved_api_key" not in st.session_state:
    st.session_state.saved_api_key = server_key

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
            value=st.session_state.get("saved_api_key", ""),
            type="password",
            placeholder="AIzaSy...",
            help="Free key from Google AI Studio (https://aistudio.google.com/)"
        )
        st.caption("✨ [Get a Free Gemini API Key at aistudio.google.com](https://aistudio.google.com/) (no credit card needed).")
        if user_key_input:
            st.session_state.saved_api_key = user_key_input.strip()
        api_key = st.session_state.get("saved_api_key", "") or server_key

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
                st.session_state.vault_load_error = "Please provide an API Key above to index the demo vault."
            else:
                initialize_vault(demo_vault_dir, "Demo AI Knowledge Vault", api_key)
                st.rerun()

    elif vault_source == "Upload Custom Vault":
        st.caption("Upload multiple `.md` files or a `.zip` archive:")
        uploaded_files = st.file_uploader(
            "Upload Notes (.md, .zip)",
            type=["zip", "md"],
            accept_multiple_files=True,
            help="Select one or multiple .md files or a zipped Obsidian vault."
        )

        if uploaded_files and st.button("📥 Process & Index Vault", use_container_width=True):
            if not api_key:
                st.session_state.vault_load_error = "Please enter an API Key to index the vault."
            else:
                temp_dir = Path("data/temp_uploaded_vault")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)

                zip_files = [f for f in uploaded_files if f.name.endswith(".zip")]
                md_files = [f for f in uploaded_files if f.name.endswith(".md")]

                if zip_files:
                    extracted_path = extract_vault_zip(zip_files[0], temp_dir)
                    initialize_vault(extracted_path, f"Uploaded Vault ({zip_files[0].name})", api_key)
                    st.rerun()
                elif md_files:
                    for f in md_files:
                        note_path = temp_dir / f.name
                        note_path.write_bytes(f.getbuffer())
                    initialize_vault(temp_dir, f"Uploaded Vault ({len(md_files)} notes)", api_key)
                    st.rerun()

    # Auto-load demo vault if API key is present and not yet loaded
    if api_key and st.session_state.vector_store is None:
        initialize_vault(demo_vault_dir, "Demo AI Knowledge Vault", api_key)

    # Status banner
    if st.session_state.get("vault_load_error"):
        st.error(st.session_state.vault_load_error)
    elif st.session_state.get("vault_load_success"):
        st.success(st.session_state.vault_load_success)

    # Vault Info Card
    st.markdown("---")
    st.subheader("📊 Vault Statistics")
    if st.session_state.vault_name:
        active_model = getattr(st.session_state, "active_model", "gemini-3.6-flash")
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

    # Reset Chat Button (only displayed when conversation is active)
    if st.session_state.messages:
        st.markdown("---")
        user_msg_count = len([m for m in st.session_state.messages if m.get("role") == "user"])
        if st.button(f"🗑️ Clear Conversation ({user_msg_count} questions)", use_container_width=True):
            st.session_state.messages = []
            st.session_state.viewing_source = None
            st.session_state.active_viewer_msg_idx = None
            st.rerun()


# --- Main Chat Area ---
st.markdown('<div class="main-header">🧠 Obsidian Vault RAG Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Query your personal Obsidian knowledge base with verifiable source citations and exact passage highlighting. Powered by Google Gemini (Free).</div>', unsafe_allow_html=True)


# Display Chat Conversation
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Deduplicated Source Buttons for Assistant Responses
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            st.markdown("##### 📚 Sources Used:")
            
            raw_chunks = msg.get("retrieved_chunks", [])
            file_chunks_map = {}
            for chunk in raw_chunks:
                if chunk.filename not in file_chunks_map:
                    file_chunks_map[chunk.filename] = []
                file_chunks_map[chunk.filename].append(chunk)
            
            cols = st.columns(max(len(file_chunks_map), 1))
            for c_idx, (fname, chunks_list) in enumerate(file_chunks_map.items()):
                with cols[c_idx]:
                    sec_count = len(chunks_list)
                    label_suffix = f" ({sec_count} sections)" if sec_count > 1 else ""
                    btn_label = f"📄 {fname}{label_suffix}"
                    
                    is_active = (st.session_state.get("active_viewer_msg_idx") == idx and 
                                 st.session_state.get("viewing_source", {}).get("filename") == fname)
                    
                    btn_type = "primary" if is_active else "secondary"
                    
                    if st.button(
                        btn_label,
                        key=f"src_btn_{idx}_{c_idx}_{fname}",
                        type=btn_type,
                        help=f"Click to inspect {fname} and highlight {sec_count} passage(s)",
                        use_container_width=True
                    ):
                        if is_active:
                            st.session_state.active_viewer_msg_idx = None
                            st.session_state.viewing_source = None
                        else:
                            st.session_state.active_viewer_msg_idx = idx
                            st.session_state.viewing_source = {
                                "filename": fname,
                                "title": chunks_list[0].title,
                                "sections": [c.heading for c in chunks_list],
                                "passages": [c.text for c in chunks_list],
                                "max_similarity": max(c.similarity_score for c in chunks_list),
                                "count": sec_count
                            }
                        st.rerun()

            # Inline Document Viewer rendered right under this specific question's answer!
            if st.session_state.get("active_viewer_msg_idx") == idx and st.session_state.get("viewing_source"):
                src_info = st.session_state.viewing_source
                st.markdown("---")
                
                col_hdr, col_btn = st.columns([5, 1])
                with col_hdr:
                    st.markdown(f"#### 📖 Source Document: `{src_info['filename']}`")
                    sections_str = ", ".join(f"`{s}`" for s in src_info.get("sections", []))
                    st.caption(f"**Sections Cited:** {sections_str} | **Passages Highlighted:** {src_info.get('count', 1)} | **Max Relevance:** {src_info.get('max_similarity', 0):.1%}")
                with col_btn:
                    if st.button("✖️ Close Viewer", key=f"close_source_viewer_{idx}", use_container_width=True):
                        st.session_state.active_viewer_msg_idx = None
                        st.session_state.viewing_source = None
                        st.rerun()

                full_doc = st.session_state.loaded_notes.get(src_info["filename"])
                if full_doc:
                    passages = src_info.get("passages", [])
                    highlighted_content, count = highlight_passages_in_markdown(
                        full_content=full_doc.content,
                        passages=passages
                    )
                    st.markdown(highlighted_content, unsafe_allow_html=True)
                else:
                    st.warning(f"Could not find full note for `{src_info['filename']}`.")
                
                st.markdown("---")


# User Question Input Handling
user_input = st.chat_input("Ask a question about your Obsidian notes...")

# Handle click from example questions
if "current_prompt" in st.session_state and st.session_state.current_prompt:
    user_input = st.session_state.current_prompt
    st.session_state.current_prompt = None

if user_input:
    # Reset active viewer so the view automatically focuses on the new question
    st.session_state.active_viewer_msg_idx = None
    st.session_state.viewing_source = None

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

        # 3. Retrieve chunks, generate answer, and rerun with stable widget keys
        with st.chat_message("assistant"):
            with st.spinner("Searching Obsidian vault and reasoning with Gemini..."):
                try:
                    search_query = contextualize_query_for_search(user_input, st.session_state.messages[:-1])
                    retrieved = st.session_state.vector_store.search(search_query, top_k=5)
                    
                    client, model_name, _ = create_client_for_key(api_key)
                    rag_res: RAGResponse = generate_rag_answer(
                        query=user_input,
                        retrieved_chunks=retrieved,
                        openai_client=client,
                        api_key=api_key,
                        model_name=model_name,
                        chat_history=st.session_state.messages
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": rag_res.answer,
                        "sources": rag_res.sources,
                        "retrieved_chunks": rag_res.retrieved_chunks
                    })
                    st.rerun()

                except Exception as e:
                    err_msg = f"❌ An error occurred: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    st.rerun()
