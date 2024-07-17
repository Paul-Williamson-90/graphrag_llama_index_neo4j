import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from src.settings import SYSTEM_SETTINGS

load_dotenv()

def llm_factory():
    return OpenAI(
        model=SYSTEM_SETTINGS.model,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=500,
    )

def embedder_factory():
    return OpenAIEmbedding(
        api_key=os.getenv("OPENAI_API_KEY"),
    )