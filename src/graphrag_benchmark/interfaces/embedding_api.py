from sentence_transformers import SentenceTransformer
import torch
from typing import List


class EmbeddingClient:
    """
    Adapter for Semantic embeddings using Sentence-Transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        if device is None:
            device = (
                "cuda:1"
                if torch.cuda.device_count() > 1
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encodes a list of sentences into embeddings."""
        return self.model.encode(texts, convert_to_tensor=True)

    def calculate_similarities(self, query: str, candidates: List[str]) -> List[float]:
        """
        Calculates cosine similarity between a single query and a list of candidates.
        Returns a list of float scores matching the order of candidates.
        """
        if not candidates:
            return []

        query_emb = self.encode([query])
        candidate_embs = self.encode(candidates)

        # sentence-transformers util has cos_sim
        from sentence_transformers.util import cos_sim

        similarities = cos_sim(query_emb, candidate_embs)[0]

        return similarities.tolist()
