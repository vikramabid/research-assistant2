from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an AI Research Assistant.

Use ONLY the context below to answer the user's question.

If the answer is present in the context, answer it clearly and completely.

Do NOT say "I don't know" if the answer is clearly present in the context.

Context:
{context}

Conversation History:
{history}

Question:
{question}

Answer:
""")