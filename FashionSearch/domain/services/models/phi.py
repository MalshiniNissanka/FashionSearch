import logging
import torch
import torch.nn as nn
from torch import Tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PseudoTokenModel(nn.Module):
    """
    Pseudo Token Generator Model with improved architecture for converting image embeddings
    into pseudo word tokens using a transformer-based module.

    Args:
        input_dim: Dimension of input features (e.g., CLIP image embeddings)
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output tokens (e.g., CLIP token embeddings)
        num_tokens: Number of pseudo-tokens to generate
        num_layers: Number of transformer encoder layers
        dropout: Dropout rate
        nhead: Number of attention heads in each transformer layer
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_tokens: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.token_queries = nn.Parameter(torch.randn(num_tokens, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self._init_parameters()
        
    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _validate_input(self, x: Tensor) -> None:
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(1)}")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            Tensor of shape [batch_size, num_tokens, output_dim]
        """
        try:
            self._validate_input(x)
            x = x.to(next(self.parameters()).dtype)
            x_proj = self.input_proj(x)
            token_inputs = x_proj.unsqueeze(1) + self.token_queries.unsqueeze(0)
            tokens = self.transformer_encoder(token_inputs.transpose(0, 1)).transpose(0, 1)
            tokens = self.output_proj(tokens)
            return tokens
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise

    @torch.no_grad()
    def generate_tokens(self, x: Tensor, temperature: float = 1.0, normalize: bool = True) -> Tensor:
        """
        Generate pseudo-tokens with optional temperature scaling.
        
        Args:
            x: Input tensor
            temperature: Sampling temperature (higher = more diverse)
            normalize: Whether to L2 normalize the output tokens
            
        Returns:
            Tensor of shape [batch_size, num_tokens, output_dim]
        """
        tokens = self.forward(x)
        if temperature != 1.0:
            tokens = tokens / temperature
        if normalize:
            tokens = torch.nn.functional.normalize(tokens, dim=-1)
        return tokens

if __name__ == "__main__":
    model = PseudoTokenModel(input_dim=512, hidden_dim=2048, output_dim=512, num_tokens=5, num_layers=3, dropout=0.1, nhead=8)
    x = torch.randn(32, 512)
    tokens = model.generate_tokens(x, temperature=0.7)
    print(f"Generated tokens shape: {tokens.shape}")
