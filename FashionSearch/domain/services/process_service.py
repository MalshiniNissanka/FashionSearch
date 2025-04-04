from pathlib import Path
import torch
from .utils.encode_with_pseudo_tokens import encode_with_pseudo_tokens
from .utils.generate_image_embeddings import process_single_image
from .utils.metrics import calculate_cosine_similarity
from .inferencing import retrieval_pipeline_with_display

class ProcessService:
    @staticmethod
    def process(image_path: str, text: str, model: str):
        """
        Process the image and text using the inferencing pipeline.
        Now we assume `image_path` is already absolute — no more forced 'uploads' prefix.
        """

        domain_path = Path("FashionSearch/domain")

        args = {
            "clip_model_name": "ViT-B/32",
            "pseudo_token_model_path": str(domain_path / "services/models" / f"token_{4 if model == 'Token 4' else 1}_model.pt"),
            "embedding_load_path": str(domain_path / "images" / "image_embeddings.pt"),
            "dataset_image_path": str(domain_path / "images" / "image_db"),
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float32,
            # Pass the absolute path along to our pipeline:
            "query_image_path": image_path,
            "text_query": text,
            "top_k": 5,
            "num_tokens": 4 if model == "Token 4" else 1
        }

        results = retrieval_pipeline_with_display(args)
        # Format results with web-accessible URLs
        return [
            {
                "image_url": f"/image_db/{image_name}.jpg",  # Use web path
                "score": float(score)
            }
            for image_name, score in results
        ]
