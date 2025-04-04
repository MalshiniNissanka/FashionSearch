import logging
from typing import Tuple
import torch
import clip
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PseudoTokenEncoder:
    """Handles encoding of text with pseudo-tokens using CLIP model."""
    
    DOLLAR_TOKEN_ID = 259  # CLIP's token ID for "$"
    
    def __init__(self, clip_model: Any):  # Changed from CLIP to Any
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.device = next(clip_model.parameters()).device
    
    def _validate_inputs(
        self, 
        text: torch.Tensor, 
        pseudo_tokens: torch.Tensor, 
        num_tokens: int
    ) -> None:
        """Validate input tensors and parameters."""
        if text.dim() != 2:
            raise ValueError(f"Expected text to be 2D, got {text.dim()}D")
        if pseudo_tokens.dim() != 3:
            raise ValueError(f"Expected pseudo_tokens to be 3D, got {pseudo_tokens.dim()}D")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
            
        batch_size = text.shape[0]
        if pseudo_tokens.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between text and pseudo_tokens")

    def _find_dollar_positions(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find positions of $ tokens in the text."""
        return (text == self.DOLLAR_TOKEN_ID).nonzero(as_tuple=True)

    def _insert_pseudo_tokens(
        self, 
        embeddings: torch.Tensor, 
        pseudo_tokens: torch.Tensor,
        positions: Tuple[torch.Tensor, torch.Tensor],
        num_tokens: int
    ) -> torch.Tensor:
        """Insert pseudo-tokens at the specified positions."""
        batch_indexes, token_indexes = positions
        for batch_idx, token_idx in zip(batch_indexes, token_indexes):
            embeddings[batch_idx, token_idx:token_idx + num_tokens] = pseudo_tokens[batch_idx]
        return embeddings

    def encode(
        self, 
        text: torch.Tensor, 
        pseudo_tokens: torch.Tensor, 
        num_tokens: int = 3
    ) -> torch.Tensor:
        """
        Encode text with pseudo-tokens using CLIP model.
        
        Args:
            text: Tokenized text [batch_size, sequence_length]
            pseudo_tokens: Token embeddings [batch_size, num_tokens, embedding_dim]
            num_tokens: Number of tokens to replace each "$" with
            
        Returns:
            Encoded text features [batch_size, embedding_dim]
            
        Example:
            >>> encoder = PseudoTokenEncoder(clip_model)
            >>> text = clip.tokenize(["Make it more $ and $"])
            >>> pseudo_tokens = torch.randn(1, 3, 512)
            >>> features = encoder.encode(text, pseudo_tokens)
        """
        try:
            self._validate_inputs(text, pseudo_tokens, num_tokens)
            
            # Get initial token embeddings
            embeddings = self.clip_model.token_embedding(text).type(self.dtype)
            
            # Replace $ tokens with pseudo-tokens
            dollar_positions = self._find_dollar_positions(text)
            embeddings = self._insert_pseudo_tokens(
                embeddings, pseudo_tokens, dollar_positions, num_tokens
            )
            
            # Add positional embeddings
            embeddings = embeddings + self.clip_model.positional_embedding.type(self.dtype)
            
            # Process through transformer
            x = self.clip_model.transformer(embeddings.permute(1, 0, 2))
            x = self.clip_model.ln_final(x.permute(1, 0, 2))
            
            # Get text features
            text_features = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            return text_features @ self.clip_model.text_projection
            
        except Exception as e:
            logger.error(f"Error encoding text with pseudo-tokens: {e}")
            raise

def encode_with_pseudo_tokens(
    clip_model: Any,  # Changed from CLIP to Any
    tokenized_text: torch.Tensor,
    pseudo_tokens: torch.Tensor,
    num_tokens: int = 3
) -> torch.Tensor:
    """
    Encode text with pseudo-tokens.
    
    Args:
        clip_model: CLIP model instance
        tokenized_text: Tokenized text input [batch_size, seq_len]
        pseudo_tokens: Generated pseudo-tokens [batch_size, num_tokens, embedding_dim]
        num_tokens: Number of pseudo-tokens to use
        
    Returns:
        Text features [batch_size, embedding_dim]
    """
    # Get initial token embeddings
    token_embeddings = clip_model.token_embedding(tokenized_text)
    
    # Find positions of "$" tokens
    dollar_positions = (tokenized_text == 259).nonzero(as_tuple=True)
    batch_indices = dollar_positions[0]
    token_indices = dollar_positions[1]
    print(dollar_positions)
    print(token_embeddings)
    
    # Insert pseudo-tokens at "$" positions
    for i in range(len(batch_indices)):
        batch_idx = batch_indices[i]
        token_idx = token_indices[i]
        if i % num_tokens == 0:  # Only process first token of each group
            token_embeddings[batch_idx, token_idx:token_idx + num_tokens] = pseudo_tokens[batch_idx]
    
    # Add positional embeddings and process through transformer
    x = token_embeddings + clip_model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x)
    
    # Get text features
    text_features = x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1)]
    text_features = text_features @ clip_model.text_projection
    
    return text_features

# Example usage
if __name__ == "__main__":
    model, _ = clip.load("ViT-B/32")
    encoder = PseudoTokenEncoder(model)
    text = clip.tokenize(["Make it more $ and $"])
    pseudo_tokens = torch.randn(1, 3, 512)
    features = encoder.encode(text, pseudo_tokens)
    print(f"Encoded features shape: {features.shape}")
