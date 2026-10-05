import os
import re
import uuid
from datetime import datetime
from html import escape
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://h-a-sourveta.onrender.com").rstrip("/")

st.set_page_config(page_title="H&A SOURVETA", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

# 1. UI STYLES
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family:'Inter',sans-serif; box-sizing:border-box; }

.stApp {
    background:linear-gradient(135deg,#f6f8fc 0%,#eef3f9 55%,#fafbfd 100%);
    color:#172033;
}

[data-testid="stHeader"] { background:transparent; }

.block-container {
    max-width:1450px;
    padding:1.2rem 2rem 7rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background:#edf2f8;
    border-right:1px solid #d8e0eb;
    overflow-x:hidden!important;
}

section[data-testid="stSidebar"] > div {
    padding:1.4rem 1rem!important;
    overflow-x:hidden!important;
}

section[data-testid="stSidebar"] * { box-sizing:border-box; }

section[data-testid="stSidebar"] .stButton {
    width:100%!important;
    min-width:0!important;
    overflow:hidden!important;
}

section[data-testid="stSidebar"] .stButton > button {
    width:100%!important;
    max-width:100%!important;
    min-width:0!important;
    overflow:hidden!important;
    white-space:nowrap!important;
    text-overflow:ellipsis!important;
}

/* BRAND */
.logo {
    display:flex;
    align-items:center;
    gap:11px;
    margin-bottom:28px;
    min-width:0;
}

.logo-mark {
    width:40px;
    height:40px;
    min-width:40px;
    border-radius:12px;
    background:linear-gradient(135deg,#6257e8,#19a9bd);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:19px;
    font-weight:800;
    color:white;
    box-shadow:0 8px 24px rgba(91,87,210,.22);
}

.logo-text {
    font-size:18px;
    font-weight:800;
    letter-spacing:.2px;
    color:#172033;
    white-space:nowrap;
}

.logo-sub {
    font-size:9px;
    color:#667085;
    letter-spacing:1.7px;
    margin-top:3px;
    white-space:nowrap;
}

.sidebar-label {
    font-size:9px;
    color:#667085;
    font-weight:700;
    letter-spacing:1.5px;
    margin:22px 0 9px;
    text-transform:uppercase;
}

/* BUTTONS */
.stButton > button {
    border:1px solid #d4dce8;
    background:#fff;
    color:#344054;
    border-radius:9px;
    font-weight:600;
    min-height:38px;
    transition:all .15s ease;
}

.stButton > button:hover {
    border-color:#665be6;
    color:#4f46c8;
    background:#f5f4ff;
    transform:translateY(-1px);
}

.new-chat button {
    background:linear-gradient(135deg,#665be6,#4f46c8)!important;
    border:0!important;
    color:white!important;
    box-shadow:0 6px 16px rgba(79,70,200,.16);
}

.new-chat button:hover {
    color:white!important;
    transform:translateY(-1px);
}

/* DOCUMENTS */
.document-card {
    background:#fff;
    border:1px solid #dce3ec;
    border-radius:10px;
    padding:9px 10px;
    margin:8px 0 5px;
}

.document-card.selected {
    background:#f2f1ff;
    border-color:#7569e8;
}

.document-name {
    font-size:10px;
    font-weight:700;
    color:#344054;
    overflow:hidden;
    white-space:nowrap;
    text-overflow:ellipsis;
}

.document-card.selected .document-name { color:#4f46c8; }

.document-meta {
    font-size:9px;
    color:#718096;
    margin-top:5px;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}

.status-dot {
    display:inline-block;
    width:6px;
    height:6px;
    border-radius:50%;
    background:#12a879;
    margin-right:6px;
}

.status-dot.pending { background:#f59e0b; }
.status-dot.offline { background:#ef4444; }

.selected-document {
    border:1px solid #d8d5ff;
    background:#f4f2ff;
    border-radius:12px;
    padding:10px 12px;
    margin:12px 0;
}

.selected-document-label {
    font-size:8px;
    color:#7168d8;
    font-weight:800;
    letter-spacing:1.3px;
    text-transform:uppercase;
}

.selected-document-name {
    font-size:11px;
    font-weight:700;
    color:#343052;
    margin-top:4px;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}

/* UPLOAD */
.upload-panel {
    width:100%;
    margin-top:14px;
    padding-top:12px;
    border-top:1px solid #d5deea;
}

div[data-testid="stFileUploader"] {
    width:100%!important;
    max-width:100%!important;
    background:#fff!important;
    border:1px dashed #b9c5d6!important;
    border-radius:12px!important;
    overflow:hidden!important;
}

div[data-testid="stFileUploader"]:hover {
    border-color:#665be6!important;
    background:#faf9ff!important;
}

div[data-testid="stFileUploader"] * { max-width:100%!important; }

[data-testid="stFileUploaderFileName"] {
    overflow:hidden!important;
    text-overflow:ellipsis!important;
    white-space:nowrap!important;
}

.upload-status {
    border:1px solid #dce3ec;
    background:#fff;
    border-radius:11px;
    padding:10px 12px;
    margin-top:10px;
    font-size:10px;
    color:#526078;
    line-height:1.6;
}

/* TOP BAR */
.topbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:4px 0 18px;
    border-bottom:1px solid #dce3ec;
    margin-bottom:25px;
}

.page-title {
    font-size:14px;
    font-weight:700;
    color:#172033;
}

.page-subtitle {
    font-size:11px;
    color:#667085;
    margin-top:4px;
}

.api-status {
    display:flex;
    align-items:center;
    gap:8px;
    border:1px solid #d8e0eb;
    background:#fff;
    padding:8px 12px;
    border-radius:20px;
    font-size:10px;
    color:#526078;
    box-shadow:0 2px 8px rgba(31,45,61,.04);
}

.api-dot {
    width:6px;
    height:6px;
    border-radius:50%;
    background:#12a879;
    box-shadow:0 0 8px rgba(18,168,121,.45);
}

.api-dot.offline {
    background:#ef4444;
    box-shadow:0 0 8px rgba(239,68,68,.35);
}

/* HERO */
.hero {
    text-align:center;
    padding:55px 20px 25px;
}

.hero-icon {
    width:66px;
    height:66px;
    margin:auto;
    border-radius:20px;
    background:linear-gradient(135deg,#f0efff,#e9f9fb);
    border:1px solid #d7d9f4;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:28px;
    color:#5d56d9;
    box-shadow:0 18px 45px rgba(78,77,170,.12);
}

.hero h1 {
    font-size:34px;
    letter-spacing:-1.5px;
    margin:20px 0 8px;
    font-weight:800;
    background:linear-gradient(90deg,#172033,#6257d9);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero p {
    font-size:12px;
    color:#667085;
    max-width:590px;
    margin:auto;
    line-height:1.7;
}

.hero-line {
    width:45px;
    height:2px;
    background:linear-gradient(90deg,#665be6,#18a6b9);
    margin:18px auto 0;
    border-radius:2px;
}

.feature-row {
    display:flex;
    gap:9px;
    justify-content:center;
    margin:24px 0;
}

.feature {
    border:1px solid #dbe2ec;
    background:#fff;
    border-radius:10px;
    padding:9px 12px;
    font-size:9px;
    color:#667085;
    box-shadow:0 2px 8px rgba(31,45,61,.035);
}

.feature b { color:#344054; }

.empty-card {
    border:1px solid #dce3ec;
    background:linear-gradient(145deg,#fff,#f7f9fc);
    border-radius:16px;
    padding:22px;
    margin:15px auto;
    max-width:900px;
    box-shadow:0 4px 14px rgba(31,45,61,.04);
}

.empty-title {
    font-size:13px;
    font-weight:700;
    color:#243047;
}

.empty-text {
    font-size:10px;
    color:#667085;
    line-height:1.6;
    margin-top:5px;
}

/* CHAT */
[data-testid="stChatMessage"] {
    background:transparent!important;
    padding:0!important;
    margin:18px 0!important;
}

[data-testid="stChatMessageContent"] { min-width:0!important; }

.chat-user-label,
.chat-assistant-label {
    font-size:9px;
    font-weight:700;
    letter-spacing:1px;
    margin-bottom:5px;
    text-transform:uppercase;
}

.chat-user-label { color:#667085; }
.chat-assistant-label { color:#5d56d9; }

.answer-card {
    background:#fff;
    border:1px solid #dce3ec;
    border-radius:14px;
    padding:15px 17px;
    box-shadow:0 3px 12px rgba(31,45,61,.04);
    font-size:13px;
    line-height:1.75;
    color:#344054;
    overflow-wrap:anywhere;
}

.user-card {
    background:#f2f1ff;
    border:1px solid #dddafe;
    border-radius:14px;
    padding:12px 15px;
    font-size:13px;
    line-height:1.65;
    color:#343052;
    overflow-wrap:anywhere;
}

.thinking-status {
    display:inline-flex;
    align-items:center;
    gap:8px;
    border:1px solid #e1e5ee;
    background:#fff;
    border-radius:10px;
    padding:8px 11px;
    font-size:10px;
    color:#667085;
    margin:4px 0 9px;
}

.thinking-dot {
    width:7px;
    height:7px;
    border-radius:50%;
    background:#665be6;
    animation:pulse 1.2s infinite ease-in-out;
}

@keyframes pulse {
    0%,100% { opacity:.35; transform:scale(.8); }
    50% { opacity:1; transform:scale(1.15); }
}

/* SOURCES */
.source-box {
    margin-top:9px;
    padding:11px 12px;
    border-radius:10px;
    background:#f7f9fc;
    border:1px solid #dce3ec;
    overflow:hidden;
}

.source-title {
    font-size:9px;
    font-weight:700;
    color:#5d56d9;
    letter-spacing:.8px;
    margin-bottom:7px;
}

.source-text {
    font-size:11px;
    color:#667085;
    line-height:1.65;
    overflow-wrap:anywhere;
}

div[data-testid="stExpander"] {
    border:1px solid #dce3ec!important;
    background:#fff!important;
    border-radius:11px!important;
    margin-top:8px;
}

div[data-testid="stExpander"] summary {
    color:#344054!important;
    font-size:11px!important;
    font-weight:600!important;
}

/* CHAT INPUT */
[data-testid="stChatInput"] {
    background:#fff!important;
    border:1px solid #cbd5e1!important;
    border-radius:15px!important;
    box-shadow:0 8px 25px rgba(31,45,61,.10)!important;
}

[data-testid="stChatInput"] > div {
    background:#fff!important;
    border-radius:15px!important;
}

[data-testid="stChatInput"] textarea {
    color:#172033!important;
    background:#fff!important;
    caret-color:#5d56d9!important;
    font-size:14px!important;
}

[data-testid="stChatInput"] textarea::placeholder { color:#98a2b3!important; }

[data-testid="stChatInput"] button {
    background:#f1efff!important;
    color:#5d56d9!important;
    border-radius:10px!important;
}

[data-testid="stChatInput"] button:hover { background:#e7e4ff!important; }

footer { display:none; }

@media(max-width:900px) {
    .feature-row { flex-wrap:wrap; }
    .hero { padding-top:35px; }
    .hero h1 { font-size:28px; }
    .block-container { padding:1rem 1rem 7rem; }
    .topbar { gap:12px; }
    .api-status { white-space:nowrap; }
}

@media(max-width:600px) {
    .topbar { align-items:flex-start; flex-direction:column; }
    .hero h1 { font-size:25px; }
    .hero p { font-size:11px; }
}
</style>
""", unsafe_allow_html=True)

# 2. SESSION STATE
if "conversation_id" not in st.session_state: st.session_state.conversation_id = uuid.uuid4().hex
if "messages" not in st.session_state: st.session_state.messages = []
if "selected_document" not in st.session_state: st.session_state.selected_document = None
if "uploader_version" not in st.session_state: st.session_state.uploader_version = 0

# 3. HELPERS
def api_get(endpoint, timeout=10):
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=timeout)
        return response.json() if response.ok else None
    except (requests.RequestException, ValueError):
        return None

def display_filename(filename):
    filename = Path(str(filename)).name
    filename = re.sub(r"^[0-9a-fA-F]{24,64}[_-]", "", filename)
    return filename.replace("_", " ").strip() or "Document"

def safe_text(value):
    return escape(str(value or ""))

def render_sources(sources):
    if not sources: return
    with st.expander(f"◈  Sources · {len(sources)}"):
        for index, source in enumerate(sources, 1):
            page = safe_text(source.get("page", source.get("page_number", "Unknown")))
            section = safe_text(source.get("section", "Document"))
            source_text = safe_text(source.get("text", source.get("document", ""))[:900])
            st.markdown(
                f'<div class="source-box">'
                f'<div class="source-title">SOURCE {index} · PAGE {page} · {section}</div>'
                f'<div class="source-text">{source_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

def reset_conversation():
    st.session_state.conversation_id = uuid.uuid4().hex
    st.session_state.messages = []

# 4. LOAD BACKEND STATE
health_data = api_get("/health")
api_online = bool(health_data)

documents_data = api_get("/documents/")
documents = documents_data.get("documents", []) if isinstance(documents_data, dict) else []

valid_ids = {doc.get("document_id") for doc in documents}
if st.session_state.selected_document and st.session_state.selected_document not in valid_ids:
    st.session_state.selected_document = None

# 5. SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="logo">
        <div class="logo-mark">✦</div>
        <div>
            <div class="logo-text">H&A SOURVETA</div>
            <div class="logo-sub">FROM SOURCE TO INSIGHT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="new-chat">', unsafe_allow_html=True)
    if st.button("＋  New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)

    with st.container(height=300, border=False):
        if documents:
            for doc in documents:
                doc_id = doc.get("document_id")
                raw_filename = doc.get("filename", "Unknown document")
                filename = display_filename(raw_filename)
                status = str(doc.get("status", "unknown"))
                active = doc_id == st.session_state.selected_document
                pending = status.lower() in {"processing", "pending", "uploaded"}

                st.markdown(
                    f'<div class="document-card {"selected" if active else ""}">'
                    f'<div class="document-name">{"● " if active else "📄 "}{safe_text(filename)}</div>'
                    f'<div class="document-meta">'
                    f'<span class="status-dot {"pending" if pending else ""}"></span>'
                    f'{"Selected · " if active else ""}{safe_text(status.title())}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                if not active:
                    if st.button("Select", key=f"doc_{doc_id}", use_container_width=True):
                        st.session_state.selected_document = doc_id
                        reset_conversation()
                        st.rerun()
                else:
                    st.button("✓ Selected", key=f"selected_{doc_id}", use_container_width=True, disabled=True)
        else:
            message = "No documents yet." if api_online else "Workspace unavailable while API is offline."
            st.markdown(f'<div style="font-size:11px;color:#687386;padding:8px 2px">{message}</div>', unsafe_allow_html=True)

    selected_document = next((doc for doc in documents if doc.get("document_id") == st.session_state.selected_document), None)

    if selected_document:
        selected_filename = display_filename(selected_document.get("filename", "Document"))
        selected_status = str(selected_document.get("status", "ready")).title()
        st.markdown(
            f'<div class="selected-document">'
            f'<div class="selected-document-label">Active document</div>'
            f'<div class="selected-document-name">📄 {safe_text(selected_filename)}</div>'
            f'<div class="document-meta"><span class="status-dot"></span>{safe_text(selected_status)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label" style="margin-top:0">Add document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_version}",
        label_visibility="collapsed",
    )

    if uploaded_file:
        clean_upload_name = display_filename(uploaded_file.name)
        file_size = uploaded_file.size / 1024 / 1024

        st.markdown(
            f'<div class="upload-status"><b>PDF ready</b><br>'
            f'📄 {safe_text(clean_upload_name)}<br>{file_size:.2f} MB</div>',
            unsafe_allow_html=True,
        )

        if st.button("⬆  Upload & process document", use_container_width=True, type="primary"):
            progress = st.empty()

            try:
                progress.markdown(
                    '<div class="upload-status">◌ Uploading PDF to SOURVETA...</div>',
                    unsafe_allow_html=True,
                )

                with st.spinner("Processing document..."):
                    response = requests.post(
                        f"{API_URL}/documents/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=300,
                    )

                if response.ok:
                    data = response.json()
                    document_id = data.get("document_id")

                    if document_id:
                        st.session_state.selected_document = document_id

                    reset_conversation()
                    st.session_state.uploader_version += 1

                    progress.markdown(
                        '<div class="upload-status">✓ PDF processed successfully. Updating workspace...</div>',
                        unsafe_allow_html=True,
                    )
                    st.success("Document added to workspace.")
                    st.rerun()
                else:
                    try:
                        detail = response.json().get("detail", "Document upload failed.")
                    except Exception:
                        detail = f"Upload failed with HTTP {response.status_code}"

                    progress.empty()
                    st.error(detail)

            except requests.Timeout:
                progress.empty()
                st.error("Upload timed out. The Render backend may still be waking up or processing the PDF.")
            except requests.RequestException as exc:
                progress.empty()
                st.error(f"Unable to reach the SOURVETA backend: {exc}")
            except Exception as exc:
                progress.empty()
                st.error(f"Unexpected upload error: {exc}")

    st.markdown('<div class="sidebar-label">System</div>', unsafe_allow_html=True)

    if api_online:
        st.markdown(
            '<div style="font-size:11px;color:#526078">'
            '<span class="status-dot"></span>API connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:11px;color:#98a2b3">'
            '<span class="status-dot offline"></span>API offline</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="font-size:9px;color:#667085;line-height:1.6;'
        'border-top:1px solid #d8e0eb;padding-top:10px;margin-top:12px">'
        '<strong>Intelligence stack</strong><br>'
        'Hybrid RAG • LangGraph • Grounded AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# 6. TOP BAR
status_text = "System operational" if api_online else "Backend unavailable"
status_class = "" if api_online else "offline"

st.markdown(
    f"""
    <div class="topbar">
        <div>
            <div class="page-title">AI Document Workspace</div>
            <div class="page-subtitle">Ask questions. Follow evidence. Discover insight.</div>
        </div>
        <div class="api-status">
            <span class="api-dot {status_class}"></span>
            {status_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_doc = next((doc for doc in documents if doc.get("document_id") == st.session_state.selected_document), None)

# 7. EMPTY / LANDING STATE
if not st.session_state.messages:
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">✦</div>
        <h1>From source to insight.</h1>
        <p>
            SOURVETA transforms complex documents into grounded answers by retrieving
            evidence, reasoning over context, and showing exactly where each answer comes from.
        </p>
        <div class="hero-line"></div>
    </div>

    <div class="feature-row">
        <div class="feature">◈ <b>Hybrid Retrieval</b></div>
        <div class="feature">◎ <b>Smart Reranking</b></div>
        <div class="feature">✓ <b>Grounded Answers</b></div>
        <div class="feature">⌁ <b>Source Citations</b></div>
    </div>
    """, unsafe_allow_html=True)

    if selected_doc:
        selected_filename = display_filename(selected_doc.get("filename", "Document"))
        st.markdown(
            f'<div class="empty-card">'
            f'<div class="empty-title">📄 {safe_text(selected_filename)}</div>'
            f'<div class="empty-text">Document selected and ready for analysis. Ask a question below to begin.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif api_online:
        st.markdown(
            '<div class="empty-card">'
            '<div class="empty-title">Select a document to begin</div>'
            '<div class="empty-text">Upload a PDF or choose an existing document from your workspace.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="empty-card">'
            '<div class="empty-title">SOURVETA backend is unavailable</div>'
            '<div class="empty-text">The Render service may be waking up. Wait a moment and refresh the application.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# 8. CONVERSATION HISTORY
for message in st.session_state.messages:
    role = message["role"]

    with st.chat_message(role):
        if role == "user":
            st.markdown('<div class="chat-user-label">YOU</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-card">{safe_text(message["content"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-assistant-label">SOURVETA</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{safe_text(message["content"])}</div>', unsafe_allow_html=True)
            render_sources(message.get("sources", []))

# 9. CHAT INPUT
query = st.chat_input("Ask anything about your document...", disabled=not api_online)

if query:
    query = query.strip()

    if not query:
        st.stop()

    if not st.session_state.selected_document:
        st.warning("Please select or upload a document first.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "time": datetime.now().strftime("%H:%M"),
    })

    with st.chat_message("user"):
        st.markdown('<div class="chat-user-label">YOU</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="user-card">{safe_text(query)}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        st.markdown('<div class="chat-assistant-label">SOURVETA</div>', unsafe_allow_html=True)

        status = st.empty()
        answer_placeholder = st.empty()

        try:
            status.markdown(
                '<div class="thinking-status">'
                '<span class="thinking-dot"></span>Searching document evidence...</div>',
                unsafe_allow_html=True,
            )

            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "query": query,
                    "document_id": st.session_state.selected_document,
                    "conversation_id": st.session_state.conversation_id,
                },
                timeout=300,
            )

            if not response.ok:
                try:
                    detail = response.json().get("detail", "Request failed.")
                except Exception:
                    detail = f"Request failed with HTTP {response.status_code}"

                status.empty()
                st.error(detail)
                st.stop()

            data = response.json()
            answer = str(data.get("answer", "")).strip()
            sources = data.get("sources", [])

            status.markdown(
                '<div class="thinking-status">'
                '<span class="thinking-dot"></span>Evidence retrieved · preparing grounded answer...</div>',
                unsafe_allow_html=True,
            )

            if answer:
                answer_placeholder.markdown(
                    f'<div class="answer-card">{safe_text(answer)}</div>',
                    unsafe_allow_html=True,
                )

                status.empty()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "time": datetime.now().strftime("%H:%M"),
                })

                render_sources(sources)
            else:
                status.empty()
                st.error("The backend returned an empty answer.")

        except requests.Timeout:
            status.empty()
            st.error("The request timed out. The Render backend may be waking up or processing the document.")
        except requests.RequestException as exc:
            status.empty()
            st.error(f"Unable to reach the SOURVETA backend: {exc}")
        except Exception as exc:
            status.empty()
            st.error(f"Unexpected error: {exc}")