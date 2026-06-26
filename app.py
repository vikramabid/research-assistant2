import tempfile
import streamlit as st

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.embeddings.hf_embeddings import get_embeddings
from src.vectorstore.faiss_store import create_vectorstore
from src.retriever.retriever import get_retriever
from src.llm.gemini import get_llm
from src.chains.rag_chain import create_rag_chain

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Assistant")
st.write("Upload a PDF and ask questions.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

        tmp.write(uploaded_file.read())

        pdf_path = tmp.name

    with st.spinner("Processing PDF..."):

        docs = load_pdf(pdf_path)

        chunks = chunk_documents(docs)

        embeddings = get_embeddings()

        vectorstore = create_vectorstore(
            chunks,
            embeddings
        )

        retriever = get_retriever(vectorstore)

        llm = get_llm()

        rag_chain = create_rag_chain(
            retriever,
            llm
        )

    st.success("PDF Indexed Successfully!")

    question = st.text_input(
        "Ask a question"
    )

    if question:

        with st.spinner("Thinking..."):

            answer = rag_chain.invoke(question)

        st.subheader("Answer")

        st.write(answer)