from src.prompts.rag_prompt import RAG_PROMPT

prompt = RAG_PROMPT.invoke(
    {
        "context": "Artificial Intelligence is the simulation of human intelligence.",
        "question": "What is AI?",
    }
)

print(prompt)