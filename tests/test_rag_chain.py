from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.embeddings.hf_embeddings import get_embeddings
from src.vectorstore.faiss_store import create_vectorstore
from src.retriever.retriever import get_retriever
from src.llm.gemini import get_llm
from src.chains.rag_chain import create_rag_chain

# Load PDF
docs = load_pdf("sample.pdf")

# Chunk
chunks = chunk_documents(docs)

# Embeddings
embeddings = get_embeddings()

# Vector Store
vectorstore = create_vectorstore(chunks, embeddings)

# Retriever
retriever = get_retriever(vectorstore)

# Gemini
llm = get_llm()

# Chain
rag_chain = create_rag_chain(retriever, llm)

# Ask Question
response = rag_chain.invoke(
    "What is Artificial Intelligence?"
)

print("=" * 60)
print(response)
print("=" * 60)