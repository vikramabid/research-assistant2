import tempfile
import shutil
import streamlit as st

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.embeddings.hf_embeddings import get_embeddings
from src.vectorstore.faiss_store import (
    create_vectorstore,
    load_vectorstore,
    save_hash,
    load_hash
)
from src.utils.hashing import calculate_pdf_hash
from src.retriever.retriever import get_retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.llm.gemini import get_llm
from src.chains.rag_chain import create_rag_chain
from src.chat.memory import (
    initialize_chat,
    display_chat,
    add_user_message,
    add_ai_message,
)

# ---------------- Streamlit Config ---------------- #

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Assistant")
st.caption(
    "Retrieval-Augmented Generation using Gemini + HuggingFace + FAISS"
)

# ---------------- Session ---------------- #

if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0

initialize_chat()

# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.header("⚙ Dashboard")

    st.write("🤖 **LLM:** Gemini 2.5 Flash")
    st.write("🧠 **Embeddings:** all-MiniLM-L6-v2")

    st.divider()

    if st.button("🗑 Clear FAISS Index"):

        shutil.rmtree(
            "data/faiss_index",
            ignore_errors=True
        )

        st.success("FAISS Index Deleted")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []
        st.session_state.chat_count = 0

        st.rerun()

display_chat()

st.write("Upload one or more PDFs and ask questions.")

# ---------------- Upload ---------------- #

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- Process PDFs ---------------- #

if uploaded_files:

    current_hash = calculate_pdf_hash(uploaded_files)
    saved_hash = load_hash()

    all_documents = []

    progress = st.progress(0)

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        docs = load_pdf(
            pdf_path,
            uploaded_file.name
        )

        all_documents.extend(docs)
        st.write(f"Loaded {uploaded_file.name}")
        st.write(f"Pages extracted: {len(docs)}")

    progress.progress(25)

    chunks = chunk_documents(all_documents)
    st.write(f"Total Documents: {len(all_documents)}")
    st.write(f"Total Chunks: {len(chunks)}")
    if len(chunks) == 0:
        st.error("No chunks were created from the uploaded PDFs.")
        st.stop()
    progress.progress(50)

    embeddings = get_embeddings()

    progress.progress(70)

    vectorstore = create_vectorstore(
    chunks,
    embeddings
)

    save_hash(current_hash)

    st.success("Created New FAISS Index")

    progress.progress(100)
    progress.empty()

    # ---------------- Retrieval ---------------- #

    retriever = get_retriever(vectorstore)

    hybrid = HybridRetriever(
        chunks,
        retriever
    )

    llm = get_llm()

    rag_chain = create_rag_chain(
        hybrid,
        llm
    )

    # ---------------- Sidebar Stats ---------------- #

    with st.sidebar:

        st.divider()

        st.subheader("📄 Uploaded PDFs")

        for pdf in uploaded_files:
            st.write(f"• {pdf.name}")

        st.metric(
            "PDFs",
            len(uploaded_files)
        )

        st.metric(
            "Pages",
            len(all_documents)
        )

        st.metric(
            "Chunks",
            len(chunks)
        )

        st.metric(
            "Questions",
            st.session_state.chat_count
        )

        st.success("✅ Vector Store Ready")

    # ---------------- Chat ---------------- #

    question = st.chat_input(
        "Ask a question..."
    )

    if question:

        add_user_message(question)

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                docs = hybrid.invoke(question)

                st.write("Retrieved Documents:", len(docs))

                for i, doc in enumerate(docs):
                    st.write(f"### Document {i+1}")
                    st.write(doc.metadata)
                    st.write(doc.page_content[:500])

                answer = rag_chain.invoke(question)

                st.markdown(answer)

        add_ai_message(answer)

        st.session_state.chat_count += 1