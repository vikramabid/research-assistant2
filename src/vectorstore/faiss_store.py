import os

from langchain_community.vectorstores import FAISS
from src.logger.logger import logger

INDEX_PATH = "data/faiss_index"

HASH_FILE = os.path.join(INDEX_PATH, "hash.txt")
def create_vectorstore(chunks, embeddings):

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(INDEX_PATH)

    return vectorstore

logger.info("FAISS index saved.")
def load_vectorstore(embeddings):

    if not os.path.exists(INDEX_PATH):

        return None

    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore
logger.info("FAISS index loaded.")
def save_hash(hash_value):

    os.makedirs(INDEX_PATH, exist_ok=True)

    with open(HASH_FILE, "w") as f:
        f.write(hash_value)


def load_hash():

    if not os.path.exists(HASH_FILE):
        return None

    with open(HASH_FILE, "r") as f:
        return f.read().strip()