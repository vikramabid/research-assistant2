import os

from dotenv import load_dotenv

load_dotenv()


class Config:

    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

    AZURE_ENDPOINT = os.getenv(
        "AZURE_OPENAI_ENDPOINT"
    )

    AZURE_API_VERSION = os.getenv(
        "AZURE_OPENAI_API_VERSION"
    )

    EMBEDDING_DEPLOYMENT = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )

    GOOGLE_API_KEY = os.getenv(
        "GOOGLE_API_KEY"
    )