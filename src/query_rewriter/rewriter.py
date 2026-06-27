from langchain_core.prompts import ChatPromptTemplate

REWRITE_PROMPT = ChatPromptTemplate.from_template(
"""
You are an AI assistant.

Given the previous conversation and the latest user question,
rewrite the latest question so that it becomes a complete,
standalone question.

Conversation:
{history}

Question:
{question}

Return ONLY the rewritten question.
"""
)


class QueryRewriter:

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, history, question):

        if not history.strip():
            return question

        prompt = REWRITE_PROMPT.invoke(
            {
                "history": history,
                "question": question,
            }
        )

        response = self.llm.invoke(prompt)

        return response.content.strip()