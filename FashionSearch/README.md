# Fashion Search Application

This is a multi-model fashion search application that uses CLIP and pseudo-tokens to find fashion items based on image and text queries.

## Setup

1. Create a virtual environment:
```bash
conda create -n fashion_search python=3.9
conda activate fashion_search
```

2. Install required packages:
```bash
pip install reflex
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install Pillow matplotlib
```

3. Setup project structure:
```bash
mkdir -p FashionSearch/domain/images/image_db
mkdir -p FashionSearch/domain/models
```

4. Run the application:
```bash
reflex run
```

## Troubleshooting

If you encounter "module 'clip' has no attribute 'load'" error, ensure you have installed the OpenAI CLIP model correctly:
```bash
pip install git+https://github.com/openai/CLIP.git
```

Make sure the model files are placed in the correct directories:
- Token models should be in `FashionSearch/domain/models/`
- Image embeddings should be in `FashionSearch/domain/images/`
- Dataset images should be in `FashionSearch/domain/images/image_db/`
