import json
import os
import uuid
from datetime import datetime

import requests
import streamlit as st
import websocket

API_URL=os.getenv("API_URL","http://127.0.0.1:8000")
WS_URL=os.getenv("WS_URL","ws://127.0.0.1:8000/ws/chat")

st.set_page_config(page_title="H&A SOURVETA",page_icon="✦",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{font-family:'Inter',sans-serif;box-sizing:border-box}
.stApp{background:linear-gradient(135deg,#f5f7fb 0%,#eef3f9 55%,#f8fafc 100%);color:#172033}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1450px;padding:1.2rem 2rem 2rem}
section[data-testid="stSidebar"]{background:#edf2f8;border-right:1px solid #d8e0eb;overflow-x:hidden!important}
section[data-testid="stSidebar"]>div{padding:1.4rem 1rem!important;overflow-x:hidden!important}
section[data-testid="stSidebar"] *{box-sizing:border-box}
section[data-testid="stSidebar"] .stButton{width:100%!important;min-width:0!important;overflow:hidden!important}
section[data-testid="stSidebar"] .stButton>button{width:100%!important;max-width:100%!important;min-width:0!important;overflow:hidden!important;white-space:nowrap!important;text-overflow:ellipsis!important}
.logo{display:flex;align-items:center;gap:11px;margin-bottom:30px;min-width:0}
.logo-mark{width:39px;height:39px;min-width:39px;border-radius:12px;background:linear-gradient(135deg,#6257e8,#19a9bd);display:flex;align-items:center;justify-content:center;font-size:19px;font-weight:800;color:white;box-shadow:0 8px 24px rgba(91,87,210,.22)}
.logo-text{font-size:18px;font-weight:800;letter-spacing:.2px;color:#172033;white-space:nowrap}
.logo-sub{font-size:9px;color:#667085;letter-spacing:1.7px;margin-top:3px;white-space:nowrap}
.sidebar-label{font-size:9px;color:#667085;font-weight:700;letter-spacing:1.5px;margin:22px 0 9px;text-transform:uppercase}
.doc-card{width:100%;max-width:100%;border:1px solid #d8e0eb;background:#fff;border-radius:12px;padding:10px;margin:6px 0;box-shadow:0 2px 8px rgba(31,45,61,.04);overflow:hidden}
.doc-card.active{border-color:#7569e8;background:#f2f1ff}
.doc-name{width:100%;max-width:100%;font-size:11px;font-weight:600;color:#243047;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.doc-meta{font-size:9px;color:#718096;margin-top:5px;white-space:nowrap}
.status-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#12a879;margin-right:6px}
.upload-panel{width:100%;margin-top:14px;padding-top:12px;border-top:1px solid #d5deea}
div[data-testid="stFileUploader"]{width:100%!important;max-width:100%!important;background:#fff!important;border:1px dashed #b9c5d6!important;border-radius:12px!important;overflow:hidden!important}
div[data-testid="stFileUploader"]:hover{border-color:#665be6!important}
div[data-testid="stFileUploader"] *{max-width:100%!important}
[data-testid="stFileUploaderFileName"]{overflow:hidden!important;text-overflow:ellipsis!important;white-space:nowrap!important}
.stButton>button{border:1px solid #d4dce8;background:#fff;color:#344054;border-radius:9px;font-weight:600}
.stButton>button:hover{border-color:#665be6;color:#4f46c8;background:#f5f4ff}
.new-chat button{background:linear-gradient(135deg,#665be6,#4f46c8)!important;border:0!important;color:white!important}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:4px 0 18px;border-bottom:1px solid #dce3ec;margin-bottom:25px}
.page-title{font-size:14px;font-weight:700;color:#172033}
.page-subtitle{font-size:11px;color:#667085;margin-top:4px}
.api-status{display:flex;align-items:center;gap:8px;border:1px solid #d8e0eb;background:#fff;padding:8px 12px;border-radius:20px;font-size:10px;color:#526078;box-shadow:0 2px 8px rgba(31,45,61,.04)}
.api-dot{width:6px;height:6px;border-radius:50%;background:#12a879;box-shadow:0 0 8px rgba(18,168,121,.45)}
.hero{text-align:center;padding:58px 20px 28px}
.hero-icon{width:66px;height:66px;margin:auto;border-radius:20px;background:linear-gradient(135deg,#f0efff,#e9f9fb);border:1px solid #d7d9f4;display:flex;align-items:center;justify-content:center;font-size:28px;color:#5d56d9;box-shadow:0 18px 45px rgba(78,77,170,.12)}
.hero h1{font-size:34px;letter-spacing:-1.5px;margin:20px 0 8px;font-weight:800;background:linear-gradient(90deg,#172033,#6257d9);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{font-size:12px;color:#667085;max-width:590px;margin:auto;line-height:1.7}
.hero-line{width:45px;height:2px;background:linear-gradient(90deg,#665be6,#18a6b9);margin:18px auto 0;border-radius:2px}
.feature-row{display:flex;gap:9px;justify-content:center;margin:24px 0}
.feature{border:1px solid #dbe2ec;background:#fff;border-radius:10px;padding:9px 12px;font-size:9px;color:#667085;box-shadow:0 2px 8px rgba(31,45,61,.035)}
.feature b{color:#344054}
[data-testid="stChatMessage"]{background:transparent!important;padding:0!important;margin:18px 0!important}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]{font-size:13px;line-height:1.75;color:#344054}
[data-testid="stChatMessageContent"]{min-width:0!important}
.source-box{margin-top:10px;padding:10px 12px;border-radius:10px;background:#f7f9fc;border:1px solid #dce3ec;overflow:hidden}
.source-title{font-size:9px;font-weight:700;color:#667085;letter-spacing:1px;margin-bottom:7px}
[data-testid="stChatInput"]{border:1px solid #cbd5e1!important;background:#fff!important;border-radius:15px!important;box-shadow:0 8px 25px rgba(31,45,61,.08)!important}
[data-testid="stChatInput"] textarea{color:#172033!important}
[data-testid="stChatInput"] textarea::placeholder{color:#98a2b3!important}
.empty-card{border:1px solid #dce3ec;background:linear-gradient(145deg,#fff,#f7f9fc);border-radius:16px;padding:22px;margin:15px auto;max-width:900px;box-shadow:0 4px 14px rgba(31,45,61,.04)}
.empty-title{font-size:13px;font-weight:700;color:#243047}
.empty-text{font-size:10px;color:#667085;line-height:1.6;margin-top:5px}
div[data-testid="stExpander"]{border:1px solid #dce3ec!important;background:#fff!important;border-radius:11px!important}
div[data-testid="stExpander"] summary{color:#344054!important}
footer{display:none}
@media(max-width:900px){.feature-row{flex-wrap:wrap}.hero h1{font-size:28px}.block-container{padding:1rem}.topbar{gap:12px}.api-status{white-space:nowrap}}
</style>
""",unsafe_allow_html=True)

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id=uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages=[]
if "selected_document" not in st.session_state:
    st.session_state.selected_document=None

def api_get(endpoint,timeout=10):
    try:
        response=requests.get(f"{API_URL}{endpoint}",timeout=timeout)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None

api_online=bool(api_get("/health"))
documents_data=api_get("/documents/")
documents=documents_data.get("documents",[]) if documents_data else []

with st.sidebar:
    st.markdown("""
    <div class="logo">
        <div class="logo-mark">✦</div>
        <div>
            <div class="logo-text">H&A SOURVETA</div>
            <div class="logo-sub">FROM SOURCE TO INSIGHT</div>
        </div>
    </div>
    """,unsafe_allow_html=True)

    st.markdown('<div class="new-chat">',unsafe_allow_html=True)
    if st.button("＋  New conversation",use_container_width=True):
        st.session_state.conversation_id=uuid.uuid4().hex
        st.session_state.messages=[]
        st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Workspace</div>',unsafe_allow_html=True)

    # Only the document list is scrollable.
    with st.container(height=380,border=False):
        if documents:
            for doc in documents:
                doc_id=doc.get("document_id")
                filename=doc.get("filename","Unknown document")
                status=doc.get("status","unknown")
                active=doc_id==st.session_state.selected_document

                if st.button(f"📄  {filename}",key=f"doc_{doc_id}",use_container_width=True):
                    st.session_state.selected_document=doc_id
                    st.rerun()

                st.markdown(
                    f'<div class="doc-card {"active" if active else ""}">'
                    f'<div class="doc-name">{"● " if active else ""}{filename}</div>'
                    f'<div class="doc-meta"><span class="status-dot"></span>{status.title()}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown('<div style="font-size:11px;color:#687386;padding:8px 2px">No documents yet.</div>',unsafe_allow_html=True)

    st.markdown('<div class="upload-panel">',unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label" style="margin-top:0">Add document</div>',unsafe_allow_html=True)

    uploaded_file=st.file_uploader("Upload PDF",type=["pdf"],label_visibility="collapsed")

    if uploaded_file and st.button("Upload to workspace",use_container_width=True):
        try:
            with st.spinner("Uploading document..."):
                response=requests.post(
                    f"{API_URL}/documents/upload",
                    files={"file":(uploaded_file.name,uploaded_file.getvalue(),"application/pdf")},
                    timeout=120
                )
            if response.ok:
                st.session_state.selected_document=response.json()["document_id"]
                st.success("Document uploaded")
                st.rerun()
            else:
                try:
                    st.error(response.json().get("detail","Upload failed."))
                except Exception:
                    st.error("Upload failed.")
        except requests.RequestException:
            st.error("FastAPI backend is unavailable.")

    st.markdown('<div class="sidebar-label">System</div>',unsafe_allow_html=True)

    if api_online:
        st.markdown('<div style="font-size:11px;color:#526078"><span class="status-dot"></span>API connected</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:11px;color:#98a2b3">○ API offline</div>',unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:9px;color:#667085;line-height:1.6;border-top:1px solid #d8e0eb;padding-top:10px;margin-top:12px">'
        '<strong>System</strong><br>Hybrid RAG • LangGraph • Grounded AI</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>',unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
    <div>
        <div class="page-title">AI Document Workspace</div>
        <div class="page-subtitle">Ask questions. Follow evidence. Discover insight.</div>
    </div>
    <div class="api-status"><span class="api-dot"></span>System operational</div>
</div>
""",unsafe_allow_html=True)

selected_doc=next((doc for doc in documents if doc.get("document_id")==st.session_state.selected_document),None)

if not st.session_state.messages:
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">✦</div>
        <h1>From source to insight.</h1>
        <p>SOURVETA transforms complex documents into grounded answers by retrieving evidence, reasoning over context, and showing exactly where the answer comes from.</p>
        <div class="hero-line"></div>
    </div>
    <div class="feature-row">
        <div class="feature">◈ <b>Hybrid Retrieval</b></div>
        <div class="feature">◎ <b>Smart Reranking</b></div>
        <div class="feature">✓ <b>Grounded Answers</b></div>
        <div class="feature">⌁ <b>Source Citations</b></div>
    </div>
    """,unsafe_allow_html=True)

    if selected_doc:
        st.markdown(
            f'<div class="empty-card"><div class="empty-title">📄 {selected_doc.get("filename","Document")}</div>'
            '<div class="empty-text">Document selected and ready for analysis. Ask a question below.</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="empty-card"><div class="empty-title">Select a document to begin</div>'
            '<div class="empty-text">Upload a PDF or select an existing document from your workspace.</div></div>',
            unsafe_allow_html=True
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources=message.get("sources",[])

        if sources:
            with st.expander(f"◈  Sources · {len(sources)}"):
                for source in sources:
                    page=source.get("page",source.get("page_number","Unknown"))
                    section=source.get("section","Document")
                    source_text=source.get("text",source.get("document",""))

                    st.markdown(
                        f'<div class="source-box"><div class="source-title">PAGE {page} · {section}</div>'
                        f'<div style="font-size:11px;color:#667085;line-height:1.6">{source_text[:500]}</div></div>',
                        unsafe_allow_html=True
                    )

query=st.chat_input("Ask anything about your document...")

if query:
    if not st.session_state.selected_document:
        st.warning("Please select or upload a document first.")
        st.stop()

    st.session_state.messages.append({"role":"user","content":query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status=st.empty()
        answer_placeholder=st.empty()

        try:
            ws=websocket.create_connection(WS_URL,timeout=300,ping_interval=30,ping_timeout=300)

            ws.send(json.dumps({
                "query":query,
                "document_id":st.session_state.selected_document,
                "conversation_id":st.session_state.conversation_id
            }))

            answer=""
            sources=[]
            citations=""

            while True:
                message=json.loads(ws.recv())
                message_type=message.get("type")

                if message_type=="status":
                    status.markdown(f'`◌` {message.get("message","Processing...")}')

                elif message_type=="resolved_query":
                    status.markdown(f'`◌` Context resolved: `{message.get("query","")}`')

                elif message_type=="token":
                    answer+=message.get("content","")
                    answer_placeholder.markdown(answer)

                elif message_type=="answer":
                    if not answer:
                        answer=message.get("answer","")
                        answer_placeholder.markdown(answer)

                elif message_type=="citations":
                    citations=message.get("content","")

                elif message_type=="sources":
                    sources=message.get("sources",[])

                elif message_type=="complete":
                    status.empty()
                    break

                elif message_type=="error":
                    status.empty()
                    st.error(message.get("message","WebSocket error."))
                    break

            ws.close()

            if answer:
                st.session_state.messages.append({
                    "role":"assistant",
                    "content":answer,
                    "sources":sources,
                    "citations":citations,
                    "time":datetime.now().strftime("%H:%M")
                })

                if citations:
                    with st.expander("⌁  Citations"):
                        st.markdown(citations)

                if sources:
                    with st.expander(f"◈  Sources · {len(sources)}"):
                        for source in sources:
                            page=source.get("page",source.get("page_number","Unknown"))
                            section=source.get("section","Document")
                            source_text=source.get("text",source.get("document",""))

                            st.markdown(
                                f'<div class="source-box"><div class="source-title">PAGE {page} · {section}</div>'
                                f'<div style="font-size:11px;color:#667085;line-height:1.6">{source_text[:500]}</div></div>',
                                unsafe_allow_html=True
                            )

        except websocket.WebSocketException as exc:
            status.empty()
            st.error(f"WebSocket connection failed: {exc}")

        except Exception as exc:
            status.empty()
            st.error(f"Unexpected error: {exc}")