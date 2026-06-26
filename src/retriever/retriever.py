from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(
    vectorstore,
    search_type: str = "similarity",
    k: int = 4,
) -> VectorStoreRetriever:
    """
    Create a retriever from FAISS vectorstore.
    """

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

    return retriever