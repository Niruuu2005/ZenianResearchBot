import logging
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger('query_bot.pinecone')

class PineconeQueryClient:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not api_key or not index_name:
            raise ValueError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def query_top_k(self, vector, top_k=10):
        try:
            results = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
            return results.matches
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return []
