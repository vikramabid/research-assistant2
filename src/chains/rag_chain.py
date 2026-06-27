from src.prompts.rag_prompt import RAG_PROMPT
from src.chat.memory import get_chat_history
from src.query_rewriter.rewriter import QueryRewriter
from src.reranker.reranker import Reranker
from src.logger.logger import logger
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.rewriter = QueryRewriter(llm)
        self.reranker = Reranker()

    def invoke(self, question):
        logger.info(f"Question: {question}")
        # Retrieve documents
        docs = self.retriever.invoke(
            rewritten_question,
            k=15
        )

        docs = self.reranker.rerank(
            rewritten_question,
            docs,
            top_k=5
        )

        # Convert to context
        context = format_docs(docs)
        history = get_chat_history()
        rewritten_question = self.rewriter.rewrite(
            history,
            question
        )

        docs = self.retriever.invoke(
            rewritten_question
        )
        # Build prompt
        prompt = RAG_PROMPT.invoke(
    {
        "history": history,
        "context": context,
        "question": rewritten_question
    }
)

        # Call LLM
        response = self.llm.invoke(prompt)

        sources = []

        seen = set()
        
        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "?"
            )

            key = (source, page)

            if key not in seen:

                seen.add(key)

                sources.append(
                    f"{source} (Page {page})"
                )

        answer = response.content
        logger.info(f"Retrieved {len(docs)} documents")
        if sources:

            answer += "\n\n📚 Sources\n"

            for src in sources:

                answer += f"\n• {src}"
        logger.info("Answer generated successfully.")
        return answer


def create_rag_chain(retriever, llm):
    return RAGChain(retriever, llm)