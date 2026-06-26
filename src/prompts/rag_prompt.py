from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are an AI Research Assistant.

Answer ONLY from the provided context.

If the answer is not in the context, say:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
)