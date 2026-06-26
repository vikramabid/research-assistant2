from langchain_community.vectorstores import FAISS


def create_vectorstore(chunks, embeddings):
    """
    Create FAISS vector store from document chunks.
    """

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore


def save_vectorstore(vectorstore, path):
    """
    Save FAISS index locally.
    """

    vectorstore.save_local(path)


def load_vectorstore(path, embeddings):
    """
    Load FAISS index.
    """

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )