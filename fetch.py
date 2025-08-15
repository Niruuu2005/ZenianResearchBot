import logging
from config import Config
from pinecone import Pinecone

logger = logging.getLogger('pinecone_query')

class PineconeQueryClient:
    def __init__(self):
        self.api_key = Config.PINECONE_API_KEY
        self.index_name = Config.PINECONE_INDEX_NAME
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)

    def query_top_k(self, embedding_vector, top_k=10):
        """
        Query Pinecone index with the given embedding vector and return top_k matches.
        
        Args:
            embedding_vector (list[float]): The vector to query with.
            top_k (int): Number of top matches to return.
        
        Returns:
            list of dict: Each dict contains id, score, and metadata of a match.
        """
        try:
            query_response = self.index.query(
                vector=embedding_vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False  # If you want to reduce response size
            )
            matches = query_response.get('matches', [])
            results = []
            for match in matches:
                results.append({
                    'id': match.get('id'),
                    'score': match.get('score'),
                    'metadata': match.get('metadata')
                })
            return results
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return []

# Example usage:
if __name__ == "__main__":
    import numpy as np
    # Suppose you have a sentence or text you want to query with
    # You need to generate its embedding vector using your local embedding model or service.
    from Bot.embedding_service import EmbeddingService

    embedding_service = EmbeddingService()
    query_text = "Hypertension"
    query_vector = embedding_service.create_embeddings(query_text)
    if not query_vector:
        raise ValueError("Failed to generate query embedding")

    pinecone_client = PineconeQueryClient()
    results = pinecone_client.query_top_k(query_vector, top_k=10)

    print("Top 10 matching Pinecone results:")
    for idx, result in enumerate(results, start=1):
        print(f"{idx}. ID: {result['id']}, Score: {result['score']}")
        print(f"   Metadata: {result['metadata']}")
