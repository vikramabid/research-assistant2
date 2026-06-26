from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents

from src.embeddings.hf_embeddings import get_embeddings

from src.vectorstore.faiss_store import (
    create_vectorstore,
    save_vectorstore,
    load_vectorstore,
)

# Load PDF
docs = load_pdf("sample.pdf")

# Chunk
chunks = chunk_documents(docs)

# Embeddings
embeddings = get_embeddings()

# Create FAISS
vectorstore = create_vectorstore(
    chunks,
    embeddings,
)

print("=" * 60)
print("FAISS Created Successfully")
print("=" * 60)

# Save Index
save_vectorstore(
    vectorstore,
    "data/faiss_index",
)

print("Index Saved")

# Load Index
db = load_vectorstore(
    "data/faiss_index",
    embeddings,
)

print("Index Loaded")

# Search
results = db.similarity_search(
    "What is Artificial Intelligence?",
    k=3,
)

print("=" * 60)

for i, doc in enumerate(results, 1):

    print(f"\nResult {i}")

    print(doc.metadata)

    print(doc.page_content[:300])