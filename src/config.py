
import os

from dotenv import load_dotenv

load_dotenv()


class Config:

    # ---------------- Gemini ---------------- #

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    GEMINI_MODEL = "gemini-2.5-flash"

    TEMPERATURE = 0

    # ---------------- Embeddings ---------------- #

    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    # ---------------- Chunking ---------------- #

    CHUNK_SIZE = 1000

    CHUNK_OVERLAP = 200

    # ---------------- Retrieval ---------------- #

    TOP_K = 5

    RETRIEVAL_K = 15

    # ---------------- Vector Store ---------------- #

    INDEX_PATH = "data/faiss_index"

    HASH_FILE = os.path.join(
        INDEX_PATH,
        "hash.txt",
    )

    # ---------------- Logging ---------------- #

    LOG_DIR = "logs"

    LOG_FILE = os.path.join(
        LOG_DIR,
        "app.log",
    )

    @staticmethod
    def validate():

        required = [
            "GOOGLE_API_KEY",
        ]

        missing = []

        for key in required:

            if not os.getenv(key):

                missing.append(key)

        if missing:

            raise ValueError(
                f"Missing environment variables: {missing}"
            )