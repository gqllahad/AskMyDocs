
import tempfile
import os
import streamlit as st

from ingest.loader import load_document
from ingest.chunker import chunk_pages
from ingest.embedder import embed_chunks, list_documents, delete_document
from query.generator import ask_stream


# Page config

st.set_page_config(
    page_title="AskMyDocs",
    page_icon="🗂️",
    layout="wide",
)

st.title("🗂️ AskMyDocs")
st.caption("Upload documents, then ask questions about them.")


# Session state init

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = [] 


# Sidebar: document management

with st.sidebar:
    st.header("Documents")

    # Upload
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Ingest uploaded files", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                for uploaded_file in uploaded_files:
                    # Save to a temporary file
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        pages = load_document(tmp_path)
                        
                        for page in pages:
                            page["metadata"]["source"] = uploaded_file.name

                        chunks = chunk_pages(pages)
                        embed_chunks(chunks)
                        st.success(f"✓ {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"✗ {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

    # Ingested documents list
    st.divider()
    st.subheader("Ingested documents")

    try:
        docs = list_documents()
        if docs:
            for doc in docs:
                col1, col2 = st.columns([4, 1])
                col1.write(f"🗂️ {doc}")
                if col2.button("✕", key=f"del_{doc}", help=f"Remove {doc}"):
                    delete_document(doc)
                    st.rerun()
        else:
            st.info("No documents ingested yet.")
    except Exception:
        st.info("No documents ingested yet.")

    # Clear
    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()


# Chat area 
# Previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if "citations" in msg and msg["citations"]:
            with st.expander(f"Sources ({len(msg['citations'])} chunks)"):
                for c in msg["citations"]:
                    st.markdown(
                        f"**{c['source']}** — page {c['page']} "
                        f"*(relevance: {c['score']})*\n\n> {c['text'][:300]}..."
                    )

if question := st.chat_input("Ask a question about your documents..."):

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            response_text = st.write_stream(
                ask_stream(question, st.session_state.chat_history)
            )
            citations = ask_stream.last_chunks

            if citations:
                with st.expander(f"Sources ({len(citations)} chunks)"):
                    for c in citations:
                        st.markdown(
                            f"**{c['source']}** — page {c['page']} "
                            f"*(relevance: {c['score']})*\n\n> {c['text'][:300]}..."
                        )

        except RuntimeError as e:
            response_text = str(e)
            citations = []
            st.error(response_text)

    # Update history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "citations": citations,
    })

    st.session_state.chat_history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_text},
    ])
