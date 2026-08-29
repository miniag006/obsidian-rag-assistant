"""
Obsidian Vault RAG Knowledge Assistant - Split-Screen Workspace Layout
- Left Column: Modern conversational AI chat with verifiable source citations
- Right Column: Dedicated live Obsidian Markdown document reader with highlighted passages
Powered by Google Gemini API & ChromaDB.
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

# Load environment variables
load_dotenv()


def get_server_api_key() -> tuple[str, str]:
    """Safely retrieves API key from secrets or environment variables."""
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
    page_title="Obsidian RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Layout styling */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 98% !important;
    }
    
    .app-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    /* Workspace Panels */
    .chat-panel {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        min-height: 75vh;
    }

    .viewer-panel {
        background: #0b0f19;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        min-height: 75vh;
        max-height: 82vh;
        overflow-y: auto;
    }

    .viewer-header {
        border-bottom: 1px solid #334155;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }

    .viewer-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    .source-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #818cf8;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 6px;
    }

    .stat-badge {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "loaded_notes" not in st.session_state:
    st.session_state.loaded_notes = {}

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
        client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        return client, "gemini-1.5-flash", "text-embedding-004"


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
            docs = load_vault(vault_path)
            notes_map = {doc.filename: doc for doc in docs}
            chunks = chunk_vault(docs)

            client, model_name, embedding_model = create_client_for_key(active_api_key)
            vector_store = VaultVectorStore(api_key=active_api_key, openai_client=client, embedding_model=embedding_model)
            indexed_count = vector_store.index_chunks(chunks)

            st.session_state.loaded_notes = notes_map
            st.session_state.vector_store = vector_store
            st.session_state.vault_name = vault_label
            st.session_state.total_chunks = indexed_count
            st.session_state.active_model = model_name
            st.session_state.active_embedding = embedding_model
            
            # Default right panel to first note if none viewed
            if docs and not st.session_state.viewing_source:
                first_doc = docs[0]
                st.session_state.viewing_source = {
                    "filename": first_doc.filename,
                    "title": first_doc.title,
                    "sections": ["Full Note"],
                    "passages": [],
                    "max_similarity": 1.0,
                    "count": 0
                }

            st.session_state.vault_load_success = f"Loaded {len(docs)} notes ({indexed_count} chunks)"
            st.session_state.vault_load_error = None
        except Exception as e:
            st.session_state.vault_load_error = f"Error loading vault: {str(e)}"
            st.session_state.vault_load_success = None


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
        st.caption("✨ [Get a Free Gemini Key](https://aistudio.google.com/) (no credit card needed).")
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
        st.caption("5 AI engineering notes: RAG, LLMs, Embeddings, Vector DBs, Agents.")
        if st.button("🚀 Load Demo Vault", use_container_width=True):
            if not api_key:
                st.session_state.vault_load_error = "Please provide an API Key above to index the demo vault."
            else:
                initialize_vault(demo_vault_dir, "Demo AI Knowledge Vault", api_key)
                st.rerun()

    elif vault_source == "Upload Custom Vault":
        st.caption("Select multiple `.md` files or a `.zip` vault:")
        uploaded_files = st.file_uploader(
            "Upload Notes (.md, .zip)",
            type=["zip", "md"],
            accept_multiple_files=True,
            help="Select one or more .md files or a zipped Obsidian vault."
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
        active_model = getattr(st.session_state, "active_model", "gemini-1.5-flash")
        active_emb = getattr(st.session_state, "active_embedding", "text-embedding-004")
        st.markdown(f"""
        <div class="stat-badge">
            <b>Active:</b> {st.session_state.vault_name}<br>
            📄 <b>Notes:</b> {len(st.session_state.loaded_notes)} &nbsp;|&nbsp; 🧩 <b>Chunks:</b> {st.session_state.total_chunks}<br>
            🤖 <b>LLM:</b> <code>{active_model}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No vault loaded yet.")

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
    if st.session_state.messages:
        st.markdown("---")
        user_msg_count = len([m for m in st.session_state.messages if m.get("role") == "user"])
        if st.button(f"🗑️ Clear Conversation ({user_msg_count})", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# --- Main Split-Screen Workspace ---
st.markdown('<div class="app-title">🧠 Obsidian Vault RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Conversational knowledge retrieval grounded in your Obsidian notes with verifiable passage highlighting.</div>', unsafe_allow_html=True)

# 2-Column Split-Screen
col_chat, col_viewer = st.columns([1.15, 1.0], gap="large")

# ==========================================
# LEFT COLUMN: CONVERSATIONAL CHAT
# ==========================================
with col_chat:
    st.markdown("#### 💬 Assistant Conversation")
    
    # Render Chat History
    if not st.session_state.messages:
        st.info("👋 Welcome! Ask any question about your indexed notes or click an example question from the sidebar.")
    else:
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Display Clean Source Buttons for Assistant Responses
                if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                    st.markdown("<div style='margin-top: 8px; font-size: 0.85rem; color: #94a3b8; font-weight: 600;'>📚 Sources Cited:</div>", unsafe_allow_html=True)
                    
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
                            
                            if st.button(
                                btn_label,
                                key=f"src_btn_{idx}_{c_idx}_{fname}",
                                help=f"Click to inspect {fname} in the Document Reader panel on the right",
                                use_container_width=True
                            ):
                                st.session_state.viewing_source = {
                                    "filename": fname,
                                    "title": chunks_list[0].title,
                                    "sections": [c.heading for c in chunks_list],
                                    "passages": [c.text for c in chunks_list],
                                    "max_similarity": max(c.similarity_score for c in chunks_list),
                                    "count": sec_count
                                }
                                st.rerun()

    # Chat Input Box
    user_input = st.chat_input("Ask a question about your Obsidian notes...")

    if "current_prompt" in st.session_state and st.session_state.current_prompt:
        user_input = st.session_state.current_prompt
        st.session_state.current_prompt = None

    if user_input:
        if not api_key:
            st.error("⚠️ Please provide a Gemini API Key in the sidebar.")
        elif not st.session_state.vector_store:
            st.error("⚠️ No vault loaded. Please click 'Load Demo Vault' in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("assistant"):
                with st.spinner("Searching Obsidian vault and synthesizing answer..."):
                    try:
                        search_query = contextualize_query_for_search(user_input, st.session_state.messages[:-1])
                        retrieved = st.session_state.vector_store.search(search_query, top_k=5)
                        
                        client, model_name, _ = create_client_for_key(api_key)
                        rag_res: RAGResponse = generate_rag_answer(
                            query=user_input,
                            retrieved_chunks=retrieved,
                            openai_client=client,
                            model_name=model_name,
                            chat_history=st.session_state.messages
                        )

                        # Auto-update the right panel to show the top cited note
                        if rag_res.sources and rag_res.retrieved_chunks:
                            top_chunk = rag_res.retrieved_chunks[0]
                            file_chunks = [c for c in rag_res.retrieved_chunks if c.filename == top_chunk.filename]
                            st.session_state.viewing_source = {
                                "filename": top_chunk.filename,
                                "title": top_chunk.title,
                                "sections": [c.heading for c in file_chunks],
                                "passages": [c.text for c in file_chunks],
                                "max_similarity": max(c.similarity_score for c in file_chunks),
                                "count": len(file_chunks)
                            }

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


# ==========================================
# RIGHT COLUMN: DEDICATED DOCUMENT READER
# ==========================================
with col_viewer:
    st.markdown("#### 📖 Obsidian Note Document Reader")
    
    if st.session_state.loaded_notes:
        # Note selector pills at top of reader
        all_note_names = list(st.session_state.loaded_notes.keys())
        selected_file = None
        
        # Pill selector row for quick note switching
        cols_nav = st.columns(min(len(all_note_names), 5) or 1)
        active_fname = st.session_state.viewing_source.get("filename") if st.session_state.viewing_source else all_note_names[0]
        
        for n_idx, fname in enumerate(all_note_names[:5]):
            with cols_nav[n_idx % len(cols_nav)]:
                is_active = (fname == active_fname)
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"📄 {fname}", key=f"nav_note_{fname}", type=btn_type, use_container_width=True):
                    doc = st.session_state.loaded_notes[fname]
                    # If this note was part of the last retrieved chunks, keep highlights, otherwise view clean
                    last_msg = st.session_state.messages[-1] if st.session_state.messages else {}
                    matching_chunks = [c for c in last_msg.get("retrieved_chunks", []) if c.filename == fname]
                    st.session_state.viewing_source = {
                        "filename": fname,
                        "title": doc.title,
                        "sections": [c.heading for c in matching_chunks] if matching_chunks else ["Full Note"],
                        "passages": [c.text for c in matching_chunks] if matching_chunks else [],
                        "max_similarity": max((c.similarity_score for c in matching_chunks), default=1.0),
                        "count": len(matching_chunks)
                    }
                    st.rerun()

        # Render Document Content
        src_info = st.session_state.viewing_source or {
            "filename": all_note_names[0],
            "title": st.session_state.loaded_notes[all_note_names[0]].title,
            "sections": ["Full Note"],
            "passages": [],
            "max_similarity": 1.0,
            "count": 0
        }

        full_doc = st.session_state.loaded_notes.get(src_info["filename"])
        
        if full_doc:
            passages = src_info.get("passages", [])
            highlight_count = src_info.get("count", len(passages))
            
            # Header card inside reader
            if highlight_count > 0:
                sections_str = ", ".join(f"`{s}`" for s in src_info.get("sections", []))
                st.markdown(f"""
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
                    <span style="color: #f59e0b; font-weight: 600; font-size: 0.85rem;">🔍 ACTIVE SOURCE CITATION</span><br>
                    <span style="color: #e2e8f0; font-size: 0.85rem;">Sections: {sections_str} &nbsp;|&nbsp; <b>{highlight_count} passage(s) highlighted</b> below</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; margin-bottom: 12px;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">📖 Viewing Full Note: <b>{src_info['filename']}</b></span>
                </div>
                """, unsafe_allow_html=True)

            # Highlight text
            if passages:
                highlighted_content, count = highlight_passages_in_markdown(
                    full_content=full_doc.content,
                    passages=passages
                )
                st.markdown(highlighted_content, unsafe_allow_html=True)
            else:
                st.markdown(full_doc.content, unsafe_allow_html=True)
        else:
            st.warning("Selected document not found in active vault.")
    else:
        st.info("👈 Please load the demo vault or upload your notes to inspect documents in the reader.")
