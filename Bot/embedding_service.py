from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self):
        # Load model once
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_embeddings(self, text: str):
        # Generate embedding vector (numpy array)
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()