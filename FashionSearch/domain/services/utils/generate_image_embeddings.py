import os
import json
import logging
from typing import List, Dict, Any, Set
import torch
import clip
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from .config import EmbeddingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_entries_for_split(config: EmbeddingConfig, split: str) -> List[Dict[str, Any]]:
    """
    Loads all entries for the specified split from JSON files.
    """
    entries = []
    for dt in config.dress_types:
        file_path = os.path.join(config.dataset_path, "captions", f"cap_{dt}_{split}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                entries.extend(data)
        else:
            logger.warning(f"File {file_path} does not exist.")
    if config.subset_limit is not None:
        entries = entries[:config.subset_limit]
    return entries

def process_single_image(image_path: str, model: Any, preprocess: Any, 
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Process a single image and return its embedding.
    """
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device).to(dtype)
        with torch.no_grad():
            emb = model.encode_image(image)
            return F.normalize(emb, dim=-1).squeeze(0)
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        raise

def process_image_batch(image_paths: List[str], model: Any, preprocess: Any,
                       device: torch.device, batch_size: int = 32) -> Dict[str, torch.Tensor]:
    """
    Process images in batches for better efficiency.
    """
    embeddings = {}
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        # Implementation for batch processing
        # ...
    return embeddings

def save_all_image_embeddings(config: EmbeddingConfig) -> None:
    """
    Process and save image embeddings for both train and val splits.
    """
    logger.info("Loading CLIP model...")
    clip_model, preprocess = clip.load(config.clip_model_name, device=config.device)
    clip_model = clip_model.eval().to(dtype=config.dtype)

    train_entries = load_entries_for_split(config, "train")
    val_entries = load_entries_for_split(config, "val")
    
    # Track train entries to skip their targets
    train_candidates: Set[str] = {entry["candidate"].strip() for entry in train_entries}
    
    # Collect all image paths to process
    image_paths = []
    for entry in train_entries + val_entries:
        candidate = entry.get("candidate", "").strip()
        if candidate:
            image_paths.append(os.path.join(config.dataset_path, "images", f"{candidate}.jpg"))
        
        # Add target images only for validation entries
        if entry.get("target") and entry["candidate"] not in train_candidates:
            target = entry["target"].strip()
            image_paths.append(os.path.join(config.dataset_path, "images", f"{target}.jpg"))

    # Process images and save embeddings
    try:
        embeddings = process_image_batch(image_paths, clip_model, preprocess, 
                                       config.device)
        torch.save({
            "embeddings": torch.stack(list(embeddings.values())),
            "names": list(embeddings.keys())
        }, config.embedding_save_path)
        logger.info(f"Saved {len(embeddings)} embeddings to {config.embedding_save_path}")
    except Exception as e:
        logger.error(f"Failed to save embeddings: {e}")
        raise

if __name__ == "__main__":
    config = EmbeddingConfig(
        clip_model_name="ViT-B/32",
        dataset_path="C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset",
        embedding_save_path="C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset/all_image_embeddings.pt",
        dress_types=["dress", "shirt", "toptee"]
    )
    save_all_image_embeddings(config)
