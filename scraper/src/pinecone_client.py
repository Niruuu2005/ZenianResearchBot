# pinecone_client.py

import logging
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from config import Config
from src.utils import create_unique_id

logger = logging.getLogger("article_scraper.pinecone")

class PineconeClient:
    def __init__(self):
        self.api_key = Config.PINECONE_API_KEY
        self.index_name = Config.PINECONE_INDEX_NAME
        self.dimension = Config.EMBEDDING_DIMENSION
        self.index = None
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        try:
            self.pc = Pinecone(api_key=self.api_key)
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name} with dimension {self.dimension}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=Config.PINECONE_ENVIRONMENT),
                )
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Successfully connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

    async def check_article_exists(self, unique_id: str) -> bool:
        try:
            result = await asyncio.to_thread(self.index.fetch, ids=[unique_id])
            result_dict = result.to_dict() if hasattr(result, "to_dict") else {}
            vectors = result_dict.get("vectors", {})
            exists = unique_id in vectors and bool(vectors[unique_id])
            logger.debug(f"Checked article existence for ID {unique_id}: {'exists' if exists else 'does not exist'}")
            return exists
        except Exception as e:
            logger.error(f"Error checking article existence for ID {unique_id}: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def store_article_embedding(self, embedding_data, article_idx, article_url):
        try:
            unique_id = create_unique_id(article_idx, article_url)
            if await self.check_article_exists(unique_id):
                logger.info(f"Article {article_idx} already exists in Pinecone with ID: {unique_id}")
                return {"success": True, "pinecone_id": unique_id}

            if len(embedding_data["embeddings"]) != self.dimension:
                logger.error(f"Embedding dimension mismatch: got {len(embedding_data['embeddings'])}, expected {self.dimension}")
                return {"success": False, "error": "Embedding dimension mismatch"}

            vector_data = {
                "id": unique_id,
                "values": embedding_data["embeddings"],
                "metadata": {
                    "link": embedding_data['metadata']['link'],
                    "summary": embedding_data['metadata']['summary'],
                    "timestamp": embedding_data['metadata']['timestamp'],
                    "title": embedding_data['metadata'].get('title', '')
                },
            }

            await asyncio.to_thread(self.index.upsert, vectors=[vector_data])
            logger.info(f"Successfully stored article {article_idx} in Pinecone with ID: {unique_id}")
            return {"success": True, "pinecone_id": unique_id}
        except Exception as e:
            logger.error(f"Error storing article {article_idx} in Pinecone: {e}")
            return {"success": False, "error": str(e)}

    async def get_index_stats(self):
        try:
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            logger.info(f"Pinecone index stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return None
