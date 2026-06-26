from src.llm.gemini import get_llm

llm = get_llm()

response = llm.invoke(
    "Explain Retrieval-Augmented Generation in two sentences."
)

print("=" * 60)

print(response.content)