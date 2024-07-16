import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

def llm_factory():
    return OpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=500,
    )

def embedder_factory():
    return OpenAIEmbedding(
        api_key=os.getenv("OPENAI_API_KEY"),
    )