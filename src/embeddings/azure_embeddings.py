import os

from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


def get_embeddings():
    """
    Return Azure OpenAI Embedding Model
    """

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )

    return embeddings