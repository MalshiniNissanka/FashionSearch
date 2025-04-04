from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class EmbeddingConfig:
    clip_model_name: str
    dataset_path: str
    embedding_save_path: str
    dress_types: List[str]
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    subset_limit: Optional[int] = None
