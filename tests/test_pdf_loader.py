from src.ingestion.pdf_loader import load_pdf

docs = load_pdf("sample.pdf")

print("=" * 60)

print(f"Pages Loaded : {len(docs)}")

print("=" * 60)

print(docs[0].metadata)

print("=" * 60)

print(docs[0].page_content[:500])