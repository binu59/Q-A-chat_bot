import os
import time
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from web_router import answer_question

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your .env file.")
    st.stop()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 0 !important; max-width: 760px; }

/* ── Top header bar ── */
.dm-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: white;
    border: 0.5px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 16px;
}
.dm-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none;
}
.dm-logo-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    background: #E6F1FB;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
}
.dm-logo-text {
    font-size: 17px;
    font-weight: 600;
    color: #111;
    letter-spacing: -0.3px;
}
.dm-logo-sub {
    font-size: 12px;
    color: #888;
    margin-left: 2px;
}
.dm-badge {
    font-size: 11px;
    background: #E6F1FB;
    color: #185FA5;
    border-radius: 20px;
    padding: 3px 10px;
    font-weight: 500;
    white-space: nowrap;
}

/* ── File attached pill ── */
.dm-file-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #E6F1FB;
    border: 0.5px solid #B5D4F4;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 13px;
    color: #185FA5;
    font-weight: 500;
    margin-bottom: 12px;
}

/* ── Empty state ── */
.dm-empty {
    text-align: center;
    padding: 48px 24px;
    color: #888;
}
.dm-empty-icon {
    font-size: 40px;
    margin-bottom: 12px;
}
.dm-empty-title {
    font-size: 17px;
    font-weight: 600;
    color: #333;
    margin-bottom: 6px;
}
.dm-empty-sub {
    font-size: 14px;
    color: #888;
    line-height: 1.6;
}

/* ── Chat messages ── */
.dm-msg-row {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 16px;
}
.dm-msg-row.user { flex-direction: row-reverse; }

.dm-avatar {
    width: 30px; height: 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}
.dm-avatar.bot { background: #E6F1FB; }
.dm-avatar.user { background: #EAF3DE; }

.dm-bubble {
    max-width: 80%;
    padding: 10px 15px;
    font-size: 14px;
    line-height: 1.7;
    color: #111;
}
.dm-bubble.bot {
    background: white;
    border: 0.5px solid #e5e7eb;
    border-radius: 2px 12px 12px 12px;
}
.dm-bubble.user {
    background: #E6F1FB;
    color: #0C447C;
    border-radius: 12px 2px 12px 12px;
    text-align: right;
}

/* ── Sources pill ── */
.dm-sources {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-top: 8px;
    font-size: 11px;
    color: #888;
    background: #f5f5f5;
    border: 0.5px solid #e5e7eb;
    border-radius: 20px;
    padding: 3px 10px;
    cursor: pointer;
}

/* ── Web search badge ── */
.dm-web-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-top: 8px;
    margin-left: 6px;
    font-size: 11px;
    color: #0C7A4B;
    background: #E7F7EF;
    border: 0.5px solid #B7E4CE;
    border-radius: 20px;
    padding: 3px 10px;
}

/* ── Suggestion chips ── */
.dm-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}
.dm-chip {
    font-size: 12px;
    padding: 5px 13px;
    border-radius: 20px;
    background: #f5f5f5;
    border: 0.5px solid #e0e0e0;
    color: #555;
    cursor: pointer;
}

/* ── Divider ── */
.dm-divider {
    border: none;
    border-top: 0.5px solid #e5e7eb;
    margin: 6px 0 16px;
}

/* ── Streamlit overrides ── */
.stChatMessage { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 0.5px solid #d1d5db !important;
    padding: 10px 18px !important;
    font-size: 14px !important;
    background: #fafafa !important;
}
[data-testid="stFileUploader"] {
    border: 0.5px dashed #d1d5db !important;
    border-radius: 12px !important;
    padding: 8px !important;
    background: #fafafa !important;
}
[data-testid="stFileUploader"] label { font-size: 13px !important; color: #666 !important; }

div[data-testid="stExpander"] {
    border: 0.5px solid #e5e7eb !important;
    border-radius: 10px !important;
    background: #fafafa !important;
}
</style>
""", unsafe_allow_html=True)


# ── RAG pipeline ───────────────────────────────────────────────────────────────
def _load_pages(tmp_path):
    try:
        return PyPDFLoader(tmp_path).load()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@st.cache_resource(show_spinner="🧠 Building knowledge base…")
def build_retriever(file_hash: str, tmp_path: str):
    pages = _load_pages(tmp_path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
    ).split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    st.session_state.page_count = len(pages)
    return retriever


def get_or_build_retriever(file_hash, tmp_path):
    return build_retriever(file_hash, tmp_path)


# ── Chat history formatting ─────────────────────────────────────────────────────
def format_chat_history(messages, max_turns: int = 4) -> str:
    """Turns the last few session messages into plain text for the router prompt."""
    recent = messages[-(max_turns * 2):]
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


# ── Safe invoke ────────────────────────────────────────────────────────────────
def safe_answer(retriever, question: str, chat_history: str, max_retries: int = 3):
    """
    Retrieves PDF context, then runs it through the Groq router
    (answer_question), which decides whether to pull in live web
    results before generating the final answer.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            docs = retriever.invoke(question)
            pdf_context = "\n\n".join(doc.page_content for doc in docs)

            result = answer_question(
                question=question,
                pdf_context=pdf_context,
                chat_history=chat_history,
            )

            sources_data = [
                {"page": doc.metadata.get("page", "?"), "content": doc.page_content}
                for doc in docs
            ]

            return result["answer"], sources_data, result["used_web_search"], result["router_reason"], None
        except Exception as e:
            last_error = e
            st.error(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    return None, [], False, "", last_error


# ── PDF state helpers ──────────────────────────────────────────────────────────
def _set_active_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    pdf_id = f"{uploaded_file.name}:{file_hash}"

    if st.session_state.get("active_pdf_id") != pdf_id:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        st.session_state.active_pdf_id = pdf_id
        st.session_state.active_pdf_name = uploaded_file.name
        st.session_state.retriever = get_or_build_retriever(file_hash, tmp_path)
        st.session_state.messages = []


def _clear_active_pdf():
    for key in ["active_pdf_id", "active_pdf_name", "retriever", "page_count", "messages"]:
        st.session_state.pop(key, None)


# ── Sync PDF state ─────────────────────────────────────────────────────────────
uploaded_file = st.session_state.get("pdf_uploader")
if uploaded_file is None:
    if "retriever" in st.session_state:
        _clear_active_pdf()
else:
    _set_active_pdf(uploaded_file)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Header ─────────────────────────────────────────────────────────────────────
has_pdf = "retriever" in st.session_state
page_count = st.session_state.get("page_count", 0)

st.markdown(f"""
<div class="dm-header">
  <div class="dm-logo">
    <div class="dm-logo-icon">🧠</div>
    <div>
      <span class="dm-logo-text">DocMind</span>
      <span class="dm-logo-sub">AI</span>
    </div>
  </div>
  {"<span class='dm-badge'>✅ " + str(page_count) + " pages indexed</span>" if has_pdf else "<span class='dm-badge' style='background:#f5f5f5;color:#888;'>No PDF attached</span>"}
</div>
""", unsafe_allow_html=True)


# ── File uploader ──────────────────────────────────────────────────────────────
with st.expander("📎 Attach a PDF", expanded=not has_pdf):
    st.file_uploader(
        "Drop your PDF here",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

if has_pdf:
    pdf_name = st.session_state.get("active_pdf_name", "document.pdf")
    st.markdown(f"""
    <div class="dm-file-pill">
      📄 {pdf_name}
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="dm-divider">', unsafe_allow_html=True)


# ── Chat area ──────────────────────────────────────────────────────────────────
if not has_pdf:
    st.markdown("""
    <div class="dm-empty">
      <div class="dm-empty-icon">🧠</div>
      <div class="dm-empty-title">Welcome to DocMind</div>
      <div class="dm-empty-sub">Upload a PDF above and start asking questions.<br>I'll read it and answer anything you want to know.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Welcome message on first load
    if not st.session_state.messages:
        pdf_name = st.session_state.get("active_pdf_name", "your PDF")
        st.markdown(f"""
        <div class="dm-msg-row">
          <div class="dm-avatar bot">🧠</div>
          <div>
            <div class="dm-bubble bot">
              PDF loaded! I've indexed <strong>{pdf_name}</strong> ({page_count} pages).<br>
              Ask me anything about it.
            </div>
            <div class="dm-chips">
              <span class="dm-chip">📝 Summarize this document</span>
              <span class="dm-chip">🔍 Key findings?</span>
              <span class="dm-chip">📌 Main topics covered?</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="dm-msg-row user">
              <div class="dm-avatar user">👤</div>
              <div class="dm-bubble user">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pages_cited = list(set(
                str(s["page"] + 1) for s in msg.get("sources", [])
                if s.get("page") not in [None, "?"]
            ))
            source_pill = ""
            if pages_cited:
                source_pill = f'<div class="dm-sources">📎 {len(pages_cited)} source(s) · page(s) {", ".join(sorted(pages_cited, key=int))}</div>'

            web_badge = ""
            if msg.get("used_web_search"):
                web_badge = '<div class="dm-web-badge">🌐 Live web results</div>'

            st.markdown(f"""
            <div class="dm-msg-row">
              <div class="dm-avatar bot">🧠</div>
              <div>
                <div class="dm-bubble bot">{msg["content"]}</div>
                {source_pill}
                {web_badge}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if msg.get("sources"):
                with st.expander("View source chunks"):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(f"**Chunk {i+1}** — page {src['page'] + 1 if src['page'] not in [None, '?'] else '?'}")
                        st.caption(src["content"])


# ── Chat input ─────────────────────────────────────────────────────────────────
if has_pdf:
    if question := st.chat_input("Ask anything about your PDF…"):

        chat_history = format_chat_history(st.session_state.messages)

        st.session_state.messages.append({"role": "user", "content": question})

        st.markdown(f"""
        <div class="dm-msg-row user">
          <div class="dm-avatar user">👤</div>
          <div class="dm-bubble user">{question}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("DocMind is thinking…"):
            answer, sources_data, used_web_search, router_reason, error = safe_answer(
                st.session_state.retriever, question, chat_history,
            )

        if error is not None:
            err_msg = "⚠️ Something went wrong after 3 retries. Please try again in a moment."
            st.warning(err_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg,
                "sources": [],
            })
        else:
            pages_cited = list(set(
                str(s["page"] + 1) for s in sources_data
                if s.get("page") not in [None, "?"]
            ))
            source_pill = ""
            if pages_cited:
                source_pill = f'<div class="dm-sources">📎 {len(pages_cited)} source(s) · page(s) {", ".join(sorted(pages_cited, key=int))}</div>'

            web_badge = ""
            if used_web_search:
                web_badge = '<div class="dm-web-badge">🌐 Live web results</div>'

            st.markdown(f"""
            <div class="dm-msg-row">
              <div class="dm-avatar bot">🧠</div>
              <div>
                <div class="dm-bubble bot">{answer}</div>
                {source_pill}
                {web_badge}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if sources_data:
                with st.expander("View source chunks"):
                    for i, src in enumerate(sources_data):
                        pg = src["page"]
                        st.markdown(f"**Chunk {i+1}** — page {pg + 1 if pg not in [None, '?'] else '?'}")
                        st.caption(src["content"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_data,
                "used_web_search": used_web_search,
            })

else:
    st.chat_input("Attach a PDF first…", disabled=True)