from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Config


def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL
    )

    return embeddings