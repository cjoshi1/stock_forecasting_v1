"""
Core FT-Transformer implementation.

A clean, optimized implementation of the Feature Tokenizer Transformer
for general tabular data tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import numpy as np


class FeatureTokenizer(nn.Module):
    """Converts tabular features into transformer tokens."""
    
    def __init__(self, num_numerical: int, cat_cardinalities: List[int], d_token: int):
        super().__init__()
        
        # Validation
        if num_numerical + len(cat_cardinalities) == 0:
            raise ValueError("Must have at least one feature")
        
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        
        # CLS token for prediction
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Numerical feature tokenization: f(x) = x * W + b
        if num_numerical > 0:
            self.num_weights = nn.Parameter(torch.randn(num_numerical, d_token))
            self.num_biases = nn.Parameter(torch.randn(num_numerical, d_token))
        
        # Categorical feature embeddings
        if cat_cardinalities:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_token) 
                for cardinality in cat_cardinalities
            ])
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_num: Numerical features [batch_size, num_numerical]
            x_cat: Categorical features [batch_size, num_categorical] 
            
        Returns:
            tokens: [batch_size, 1 + num_features, d_token]
        """
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        tokens = []
        
        # Process numerical features
        if self.num_numerical > 0 and x_num is not None:
            # Broadcast: [batch, num_features, 1] * [1, num_features, d_token] + [1, num_features, d_token]
            num_tokens = (x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + 
                         self.num_biases.unsqueeze(0))
            tokens.append(num_tokens)
        
        # Process categorical features  
        if self.cat_cardinalities and x_cat is not None:
            cat_tokens = torch.stack([
                self.cat_embeddings[i](x_cat[:, i]) 
                for i in range(len(self.cat_cardinalities))
            ], dim=1)
            tokens.append(cat_tokens)
        
        # Combine all feature tokens
        if not tokens:
            raise ValueError("No valid input features provided")
        
        feature_tokens = torch.cat(tokens, dim=1)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, feature_tokens], dim=1)


class SequenceFeatureTokenizer(nn.Module):
    """Tokenizes sequences of tabular features for temporal FT-Transformer."""
    
    def __init__(self, num_numerical: int, cat_cardinalities: List[int], d_token: int, sequence_length: int):
        super().__init__()
        
        # Validation
        if num_numerical + len(cat_cardinalities) == 0:
            raise ValueError("Must have at least one feature")
        
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        self.sequence_length = sequence_length
        
        # CLS token for prediction
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Numerical feature tokenization: f(x) = x * W + b (same for all timesteps)
        if num_numerical > 0:
            self.num_weights = nn.Parameter(torch.randn(num_numerical, d_token))
            self.num_biases = nn.Parameter(torch.randn(num_numerical, d_token))
        
        # Categorical feature embeddings (same for all timesteps)
        if cat_cardinalities:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_token) 
                for cardinality in cat_cardinalities
            ])
        
        # Temporal positional encoding - learnable position embeddings for each timestep
        self.temporal_pos_embedding = nn.Parameter(torch.randn(sequence_length, d_token))
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Sequential features [batch_size, sequence_length, num_features]
            
        Returns:
            tokens: [batch_size, 1 + sequence_length * tokens_per_timestep, d_token]
        """
        batch_size, seq_len, total_features = x_seq.shape
        
        # Split into numerical and categorical features
        if self.num_numerical > 0 and len(self.cat_cardinalities) > 0:
            x_num_seq = x_seq[:, :, :self.num_numerical]  # [batch, seq_len, num_numerical]  
            x_cat_seq = x_seq[:, :, self.num_numerical:]  # [batch, seq_len, num_categorical]
        elif self.num_numerical > 0:
            x_num_seq = x_seq
            x_cat_seq = None
        else:
            x_num_seq = None
            x_cat_seq = x_seq
            
        all_tokens = []
        
        # Process each timestep
        for t in range(seq_len):
            timestep_tokens = []
            
            # Process numerical features for this timestep
            if self.num_numerical > 0 and x_num_seq is not None:
                x_num_t = x_num_seq[:, t, :]  # [batch, num_numerical]
                # Apply tokenization: value * weight + bias
                num_tokens = (x_num_t.unsqueeze(-1) * self.num_weights.unsqueeze(0) + 
                             self.num_biases.unsqueeze(0))  # [batch, num_numerical, d_token]
                timestep_tokens.append(num_tokens)
            
            # Process categorical features for this timestep  
            if self.cat_cardinalities and x_cat_seq is not None:
                x_cat_t = x_cat_seq[:, t, :].long()  # [batch, num_categorical]
                cat_tokens = torch.stack([
                    self.cat_embeddings[i](x_cat_t[:, i]) 
                    for i in range(len(self.cat_cardinalities))
                ], dim=1)  # [batch, num_categorical, d_token]
                timestep_tokens.append(cat_tokens)
            
            # Combine tokens for this timestep
            if timestep_tokens:
                timestep_combined = torch.cat(timestep_tokens, dim=1)  # [batch, tokens_per_timestep, d_token]
                
                # Add temporal positional encoding to all tokens of this timestep
                timestep_combined = timestep_combined + self.temporal_pos_embedding[t].unsqueeze(0).unsqueeze(0)
                
                all_tokens.append(timestep_combined)
        
        # Concatenate all timesteps: [batch, seq_len * tokens_per_timestep, d_token]
        if all_tokens:
            sequence_tokens = torch.cat(all_tokens, dim=1)
        else:
            # Fallback (shouldn't happen with validation)
            device = self.cls_token.device
            sequence_tokens = torch.empty(batch_size, 0, self.d_token, device=device)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, sequence_tokens], dim=1)


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data."""
    
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        if d_ffn is None:
            d_ffn = 4 * d_token
            
        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(num_numerical, cat_cardinalities, d_token)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head will be added by wrapper based on task
        self.d_token = d_token
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_num: Numerical features [batch_size, num_numerical]
            x_cat: Categorical features [batch_size, num_categorical]
            
        Returns:
            cls_output: [batch_size, d_token] - CLS token representation
        """
        # Tokenize features
        tokens = self.tokenizer(x_num, x_cat)
        
        # Pass through transformer
        transformer_out = self.transformer(tokens)
        
        # Return CLS token output
        return transformer_out[:, 0, :]  # [batch_size, d_token]


class SequenceFTTransformer(nn.Module):
    """Feature Tokenizer Transformer for sequential tabular data."""
    
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        sequence_length: int,
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        if d_ffn is None:
            d_ffn = 4 * d_token
            
        # Sequence feature tokenizer
        self.tokenizer = SequenceFeatureTokenizer(num_numerical, cat_cardinalities, d_token, sequence_length)
        
        # Transformer encoder - same as before, but will process more tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head will be added by wrapper based on task
        self.d_token = d_token
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Sequential features [batch_size, sequence_length, num_features]
            
        Returns:
            cls_output: [batch_size, d_token] - CLS token representation
        """
        # Tokenize sequence features  
        tokens = self.tokenizer(x_seq)
        
        # Pass through transformer
        transformer_out = self.transformer(tokens)
        
        # Return CLS token output
        return transformer_out[:, 0, :]  # [batch_size, d_token]


class FTTransformerPredictor(nn.Module):
    """Complete FT-Transformer with prediction head."""
    
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int], 
        n_classes: int = 1,
        **transformer_kwargs
    ):
        super().__init__()
        
        self.ft_transformer = FTTransformer(num_numerical, cat_cardinalities, **transformer_kwargs)
        
        # Prediction head
        d_token = self.ft_transformer.d_token
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(transformer_kwargs.get('dropout', 0.1)),
            nn.Linear(d_token, n_classes)
        )
        
        self.n_classes = n_classes
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_num: Numerical features
            x_cat: Categorical features  
            
        Returns:
            predictions: [batch_size, n_classes]
        """
        cls_output = self.ft_transformer(x_num, x_cat)
        return self.head(cls_output)


class SequenceFTTransformerPredictor(nn.Module):
    """Complete Sequential FT-Transformer with prediction head."""
    
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int], 
        sequence_length: int,
        n_classes: int = 1,
        **transformer_kwargs
    ):
        super().__init__()
        
        self.sequence_ft_transformer = SequenceFTTransformer(
            num_numerical, cat_cardinalities, sequence_length, **transformer_kwargs
        )
        
        # Prediction head
        d_token = self.sequence_ft_transformer.d_token
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(transformer_kwargs.get('dropout', 0.1)),
            nn.Linear(d_token, n_classes)
        )
        
        self.n_classes = n_classes
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Sequential features [batch_size, sequence_length, num_features]
            
        Returns:
            predictions: [batch_size, n_classes]
        """
        cls_output = self.sequence_ft_transformer(x_seq)
        return self.head(cls_output)