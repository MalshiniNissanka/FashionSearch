import json
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal, TypedDict
import torch
from PIL import Image
from torch.utils.data import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageItem(TypedDict):
    image: torch.Tensor
    image_name: str

class RelativeItem(TypedDict):
    reference_image: torch.Tensor
    reference_name: str
    target_image: torch.Tensor
    target_name: str
    relative_captions: List[str]

class FashionIQDataset(Dataset):
    """FashionIQ dataset for fashion image retrieval"""

    VALID_SPLITS = ['train', 'val']
    VALID_MODES = ['relative', 'classic']
    VALID_DRESS_TYPES = ['dress', 'shirt', 'toptee']

    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: Literal['train', 'val'],
        dress_types: List[str],
        mode: Literal['relative', 'classic'],
        preprocess: callable,
        no_duplicates: bool = False,
        subset_limit: Optional[int] = None
    ) -> None:
        """Initialize FashionIQ dataset"""
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        
        self._validate_inputs(split, mode, dress_types)
        self.triplets = self._load_triplets(dress_types, split, no_duplicates, subset_limit)
        
        logger.info(
            f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized "
            f"with {len(self.triplets)} items"
        )

    def _validate_inputs(self, split: str, mode: str, dress_types: List[str]) -> None:
        """Validate input parameters"""
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}")
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        if not all(dt in self.VALID_DRESS_TYPES for dt in dress_types):
            raise ValueError(f"dress_types must be from {self.VALID_DRESS_TYPES}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist")

    def _load_triplets(
        self, 
        dress_types: List[str], 
        split: str, 
        no_duplicates: bool,
        subset_limit: Optional[int]
    ) -> List[Dict]:
        """Load and process triplets from caption files"""
        triplets = []
        for dress_type in dress_types:
            caption_file = self.dataset_path / 'captions' / f'cap_{dress_type}_{split}.json'
            if not caption_file.exists():
                logger.warning(f"Caption file {caption_file} not found. Skipping.")
                continue
            
            with open(caption_file, 'r') as f:
                data = json.load(f)
                triplets.extend(data)

        if no_duplicates:
            triplets = self._remove_duplicates(triplets)
        
        if subset_limit is not None:
            triplets = triplets[:subset_limit]
            
        return triplets

    @staticmethod
    def _remove_duplicates(triplets: List[Dict]) -> List[Dict]:
        """Remove duplicate entries based on candidate image"""
        seen = set()
        unique_triplets = []
        for triplet in triplets:
            if triplet['candidate'] not in seen:
                seen.add(triplet['candidate'])
                unique_triplets.append(triplet)
        return unique_triplets

    def _load_image(self, image_name: str) -> torch.Tensor:
        """Load and preprocess an image"""
        image_path = self.dataset_path / 'images' / f"{image_name}.jpg"
        try:
            with Image.open(image_path) as img:
                return self.preprocess(img)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def __getitem__(self, index: int) -> Union[ImageItem, RelativeItem]:
        """Get a dataset item"""
        try:
            triplet = self.triplets[index]
            
            if self.mode == 'classic':
                return ImageItem(
                    image=self._load_image(triplet['candidate']),
                    image_name=triplet['candidate']
                )
            
            # relative mode
            return RelativeItem(
                reference_image=self._load_image(triplet['candidate']),
                reference_name=triplet['candidate'],
                target_image=self._load_image(triplet['target']),
                target_name=triplet['target'],
                relative_captions=triplet['captions']
            )

        except Exception as e:
            logger.error(f"Error getting item at index {index}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.triplets)
