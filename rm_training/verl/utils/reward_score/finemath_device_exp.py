import torch
import numpy as np
import ray
from transformers import AutoTokenizer, AutoModel

# Initialize Ray earlier in your workflow, if not already initialized.
# ray.init()

@ray.remote(num_gpus=0.5)
class EmbeddingModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, text):
        with torch.no_grad():
            task = "text-matching"
            embedding = self.model.encode([text], task=task, device=self.device)[0]
        return embedding


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Instantiate the model actor once at the module level
_embedding_model_actor = EmbeddingModel.remote()


def compute_score(solution_str, ground_truth):
    """Compute reward score based on embedding similarity between solution and ground truth.

    Args:
        solution_str: The model's generated solution
        ground_truth: The ground truth answer

    Returns:
        float: A score between 0 and 1 based on embedding similarity
    """
    # Get embeddings remotely
    solution_embedding, ground_truth_embedding = ray.get([
        _embedding_model_actor.get_embedding.remote(solution_str),
        _embedding_model_actor.get_embedding.remote(ground_truth)
    ])

    # Compute cosine similarity
    similarity = cosine_similarity(solution_embedding, ground_truth_embedding)

    # Scale similarity from [-1, 1] to [0, 1]
    return float((similarity + 1) / 2)
