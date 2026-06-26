from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.embeddings.hf_embeddings import get_embeddings
from src.vectorstore.faiss_store import create_vectorstore
from src.retriever.retriever import get_retriever

# Load PDF
docs = load_pdf("sample.pdf")

# Chunk
chunks = chunk_documents(docs)

# Embeddings
embeddings = get_embeddings()

# Vector Store
vectorstore = create_vectorstore(
    chunks,
    embeddings,
)

# Retriever
retriever = get_retriever(
    vectorstore,
    k=3,
)

# Query
results = retriever.invoke(
    "What is Artificial Intelligence?"
)

print("=" * 60)

print(f"Retrieved {len(results)} Documents")

print("=" * 60)

for i, doc in enumerate(results, 1):

    print(f"\nResult {i}")

    print(doc.metadata)

    print("-" * 40)

    print(doc.page_content[:300])