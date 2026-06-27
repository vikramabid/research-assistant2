import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config
load_dotenv()


def get_llm():
    """
    Returns Gemini LLM.
    """

    llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=Config.TEMPERATURE,
    )

    return llm