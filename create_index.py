import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "langchain-docs-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=2048,  # Ajusta este número a la dimensión de tu modelo de Ollama
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )