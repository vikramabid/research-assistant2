from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents

docs = load_pdf("sample.pdf")

chunks = chunk_documents(docs)

print("=" * 60)

print(f"Pages : {len(docs)}")

print(f"Chunks : {len(chunks)}")

print("=" * 60)

print(chunks[0].metadata)

print("=" * 60)

print(chunks[0].page_content[:500])