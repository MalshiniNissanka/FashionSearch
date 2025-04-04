import clip
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from .utils.encode_with_pseudo_tokens import encode_with_pseudo_tokens  # Changed import
from .models.phi import PseudoTokenModel
import os
import logging

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def retrieval_pipeline_with_display(args):
    """
    Retrieval phase implementation using pre-calculated embeddings and displaying the retrieved images.
    Returns a list of tuples containing (image_name, similarity_score).
    """
    # Step 1: Load CLIP model
    clip_model, clip_preprocess = clip.load(args["clip_model_name"], device=args["device"])
    clip_model = clip_model.eval().to(dtype=args["dtype"])
    print(f"Loaded CLIP model: {args['clip_model_name']} on {args['device']}")

    # Step 2: Load pre-trained pseudo-token model with transformer architecture
    pseudo_token_model = PseudoTokenModel(
        input_dim=clip_model.visual.output_dim,
        hidden_dim=clip_model.visual.output_dim * 4,
        output_dim=clip_model.token_embedding.embedding_dim,
        num_tokens=args["num_tokens"],
        num_layers=3,
        dropout=0.1,
        nhead=8
    )
    checkpoint = torch.load(args["pseudo_token_model_path"], map_location=args["device"])
    pseudo_token_model.load_state_dict(checkpoint["model_state_dict"])
    pseudo_token_model = pseudo_token_model.eval().to(args["device"]).to(args["dtype"])
    print(f"Loaded pseudo-token model from {args['pseudo_token_model_path']}")

    # Step 3: Load pre-calculated embeddings
    embedding_data = torch.load(args["embedding_load_path"], map_location=args["device"])
    image_embeddings = embedding_data["embeddings"].to(args["device"]).to(args["dtype"])
    image_names = embedding_data["names"]
    print(f"Loaded {len(image_embeddings)} pre-calculated image embeddings.")

    # Step 4: Encode query image (if provided) and generate pseudo-tokens from it.
    if args.get("query_image_path"):
        query_image = clip_preprocess(Image.open(args["query_image_path"])).unsqueeze(0).to(args["device"]).to(args["dtype"])
        with torch.no_grad():
            query_image_embedding = clip_model.encode_image(query_image)
            query_image_embedding = F.normalize(query_image_embedding, dim=-1)
            # Generate pseudo-tokens from the query image embedding
            pseudo_tokens = pseudo_token_model(query_image_embedding)
            pseudo_tokens = F.normalize(pseudo_tokens, dim=-1)
        print(f'Query image embedding and pseudo-tokens calculated {pseudo_tokens}.')
    else:
        # If no query image is provided, we cannot generate pseudo-tokens; we use a dummy token (or skip fusion).
        pseudo_tokens = None

    # Step 5: Encode query text (if provided)
    if args.get("text_query"):
        tokenized_query = clip.tokenize([args["text_query"]]).to(args["device"])
        with torch.no_grad():
            # Use pseudo_tokens from the query image if available; otherwise, we call encode_with_pseudo_tokens without pseudo-tokens.
            if pseudo_tokens is not None:
                # Build a prompt with the appropriate number of "$" placeholders.
                num_tokens = args.get("num_tokens", 3)
                # prompt_template = "a photo of " + " ".join(["$"] * num_tokens)
                
                prompt_template =  "An image of " + " ".join(["$"]*num_tokens) + " with features " + f"{args['text_query']} "

                print(prompt_template)
                tokenized_query = clip.tokenize([prompt_template]).to(args["device"])
                print(tokenized_query)
                text_embedding = encode_with_pseudo_tokens(clip_model, tokenized_query, pseudo_tokens, num_tokens=num_tokens)
            else:
                # If no pseudo-tokens, just encode the text query directly.
                text_embedding = clip_model.encode_text(tokenized_query)
            text_embedding = F.normalize(text_embedding, dim=-1)

    # Step 6: Calculate cosine similarity and retrieve results
    similarities = torch.matmul(image_embeddings, text_embedding.T).squeeze()
    top_k_indices = torch.topk(similarities, k=args["top_k"], largest=True).indices

    # Step 7: Format results
    results = []
    for idx in top_k_indices:
        image_name = image_names[idx]
        similarity_score = similarities[idx].item()
        results.append((image_name, similarity_score))

    return results

# Remove or modify the display_images function since we'll handle display in the UI

if __name__ == "__main__":
    args = {
        "clip_model_name": "ViT-B/32",
        "pseudo_token_model_path": "C:/Users/Admin/Downloads/Malshini/MSC/MSC/token_3_model.pt",
        "embedding_load_path": "C:/Users/Admin/Downloads/Malshini/MSC/MSC/image_embeddings.pt",  # Path to pre-calculated embeddings
        "dataset_image_path": "C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset/images",         # Path to dataset images
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "dtype": torch.float32,
        "query_image_path": "C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset/images/B00EV1B9C2.jpg",  # Query image path (if any)
        "text_query": "this in white",                  # Query text (if any)
        "top_k": 5,                                             # Number of top results to retrieve
        "num_tokens": 3                                         # Number of pseudo-tokens to use
    }
    results = retrieval_pipeline_with_display(args)
    for image_name, similarity_score in results:
        print(f"Image: {image_name}, Similarity: {similarity_score:.4f}")
