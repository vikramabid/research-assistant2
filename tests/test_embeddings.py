from src.embeddings.hf_embeddings import get_embeddings

embeddings = get_embeddings()

vector = embeddings.embed_query(
    "What is Artificial Intelligence?"
)

print("=" * 60)
print("Embedding Successful")
print("=" * 60)
print("Vector Dimension:", len(vector))
print("=" * 60)
print(vector[:10])