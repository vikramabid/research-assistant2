from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.prompts.rag_prompt import RAG_PROMPT


def format_docs(docs):
    """
    Convert retrieved documents into a single context string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, llm):
    """
    Build the complete LCEL RAG chain.
    """

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain