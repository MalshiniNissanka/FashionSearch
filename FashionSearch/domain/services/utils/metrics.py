# calculate_cosine_similarity.py

import logging
from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_cosine_similarity(
    image_features: Tensor,
    text_features: Tensor,
    epsilon: float = 1e-8
) -> Tensor:
    """
    Compute cosine similarity with improved numerical stability.
    
    Args:
        image_features: Image embeddings [batch_size, feature_dim]
        text_features: Text embeddings [batch_size, feature_dim]
        epsilon: Small value for numerical stability
        
    Returns:
        Similarity matrix [batch_size, batch_size]
    """
    # Normalize features
    image_features = F.normalize(image_features, dim=-1, eps=epsilon)
    text_features = F.normalize(text_features, dim=-1, eps=epsilon)
    
    # Compute similarity
    similarity = torch.mm(image_features, text_features.T)
    similarity = torch.clamp(similarity, min=-1.0, max=1.0)
    
    # Handle NaN values
    similarity = torch.nan_to_num(similarity, nan=0.0)
    
    return similarity

class RetrievalMetrics:
    """Handles computation of various retrieval metrics for image-text pairs."""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def _validate_inputs(
        self,
        features1: Tensor,
        features2: Tensor,
        batch_dim: int = 0
    ) -> None:
        """Validate input tensors."""
        if features1.dim() != 2 or features2.dim() != 2:
            raise ValueError("Input features must be 2-dimensional")
        if features1.size(batch_dim) != features2.size(batch_dim):
            raise ValueError("Batch sizes must match")
    
    def compute_recall_at_k(
        self,
        similarity_matrix: Tensor,
        ground_truth_indices: Tensor,
        k: int
    ) -> Tuple[float, int]:
        """
        Compute Recall@K with additional statistics.
        
        Returns:
            Tuple of (recall_score, successful_retrievals)
        """
        try:
            total_queries = similarity_matrix.size(0)
            if k > total_queries:
                raise ValueError(f"k ({k}) cannot be larger than batch size ({total_queries})")
                
            # Get top-k indices
            _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
            
            # Check if ground truth is in top-k
            correct = torch.any(top_k_indices == ground_truth_indices.unsqueeze(1), dim=1)
            successful_retrievals = correct.sum().item()
            recall_score = successful_retrievals / total_queries
            
            return recall_score, successful_retrievals
            
        except Exception as e:
            logger.error(f"Error computing Recall@{k}: {e}")
            raise

    def compute_mrr(
        self,
        similarity_matrix: Tensor,
        ground_truth_indices: Tensor,
        batch_processing: bool = True
    ) -> float:
        """
        Compute Mean Reciprocal Rank with batch processing option.
        """
        try:
            if batch_processing:
                # Efficient batch computation
                ranks = torch.argsort(
                    torch.argsort(similarity_matrix, dim=1, descending=True),
                    dim=1
                )
                ranks = ranks[torch.arange(ranks.size(0)), ground_truth_indices] + 1
                mrr = (1.0 / ranks.float()).mean().item()
            else:
                # Original per-sample computation
                reciprocal_ranks = []
                for i in range(similarity_matrix.size(0)):
                    sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
                    rank = (sorted_indices == ground_truth_indices[i]).nonzero(as_tuple=True)[0].item() + 1
                    reciprocal_ranks.append(1.0 / rank)
                mrr = np.mean(reciprocal_ranks)
                
            return mrr
            
        except Exception as e:
            logger.error(f"Error computing MRR: {e}")
            raise

    def compute_similarity_stats(
        self,
        similarity_matrix: Tensor
    ) -> Tuple[float, float, float]:
        """
        Compute comprehensive similarity statistics.
        
        Returns:
            Tuple of (mean_similarity, std_deviation, median_similarity)
        """
        try:
            diag = similarity_matrix.diag()
            valid_mask = ~torch.isnan(diag)
            valid_similarities = diag[valid_mask]
            
            if len(valid_similarities) == 0:
                raise ValueError("No valid similarities found")
                
            mean = valid_similarities.mean().item()
            std = valid_similarities.std().item()
            median = valid_similarities.median().item()
            
            return mean, std, median
            
        except Exception as e:
            logger.error(f"Error computing similarity stats: {e}")
            raise

# Example usage
if __name__ == "__main__":
    metrics = RetrievalMetrics()
    # Create sample features
    img_feats = torch.randn(10, 512)
    txt_feats = torch.randn(10, 512)
    
    # Compute metrics
    similarity = calculate_cosine_similarity(img_feats, txt_feats)
    recall, count = metrics.compute_recall_at_k(similarity, torch.arange(10), k=5)
    mrr = metrics.compute_mrr(similarity, torch.arange(10))
    mean, std, median = metrics.compute_similarity_stats(similarity)
    
    print(f"Recall@5: {recall:.3f} ({count} successful)")
    print(f"MRR: {mrr:.3f}")
    print(f"Stats - Mean: {mean:.3f}, Std: {std:.3f}, Median: {median:.3f}")
