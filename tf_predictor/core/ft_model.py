"""
Core FT-Transformer implementation.

🧠 FT-TRANSFORMER CLS MODEL (UNIFIED TOKEN ARCHITECTURE)
=========================================================

The FT-Transformer CLS Model is a unified architecture for time series forecasting that
processes STATIC categorical features and TIME-VARYING numerical sequences together through
a single transformer. Unlike dual-path approaches, all features are tokenized and attend
to each other in a shared attention mechanism.

📊 ARCHITECTURE OVERVIEW:
┌─────────────────────────────────────────────────────────────────────────────┐
│  STATIC Categorical + TIME-VARYING Numerical → Unified Tokenization →       │
│  [batch, num_cat] + [batch, seq, num_num]   → [CLS, num_tokens, cat_tokens]│
│                                               ↓                              │
│                                        Shared Transformer                    │
│                                               ↓                              │
│                                          CLS Token → Prediction              │
└─────────────────────────────────────────────────────────────────────────────┘

🔄 DATA FLOW WITH MATRIX DIMENSIONS:

Example Configuration:
- batch_size = 32
- sequence_length = 10
- num_numerical = 8 (price, volume, technical indicators, etc.)
- num_categorical = 2 (symbol, sector)
- cat_cardinalities = [100, 5] (100 stock symbols, 5 sectors)
- d_model = 128 (embedding dimension)
- num_heads = 4
- num_layers = 3
- output_dim = 1 (single-step forecast)

🏗️ UNIFIED TOKENIZATION & PROCESSING:

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: INPUT FORMAT                                                        │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ Numerical Sequences (Time-Varying):                                         │
│   x_num: [32, 10, 8]  # [batch_size, sequence_length, num_numerical]      │
│   Represents historical price data across 10 timesteps:                     │
│     - open, high, low, close, volume                                        │
│     - technical indicators (RSI, MACD, SMA)                                 │
│                                                                              │
│ Categorical Features (Static):                                              │
│   x_cat: [32, 2]  # [batch_size, num_categorical]                          │
│   Values are integer indices (label encoded):                               │
│     - Column 0: symbol indices [0-99]                                       │
│     - Column 1: sector indices [0-4]                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: NUMERICAL TOKENIZATION (TIME-VARYING FEATURES)                      │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ For each timestep t and each numerical feature j:                           │
│   token_tj = x_num[:, t, j] * W_j + b_j                                    │
│                                                                              │
│ NumericalTokenizer: Creates per-feature linear transformations              │
│   W_j: [1, d_model] for each numerical feature j                           │
│   b_j: [d_model] for each numerical feature j                              │
│                                                                              │
│ Process each timestep:                                                       │
│   for t in range(10):                                                        │
│     x_num_t = x_num[:, t, :]  # [32, 8]                                    │
│     tokens_t = NumericalTokenizer(x_num_t)  # [32, 8, 128]                 │
│     tokens_t += temporal_pos_encoding[:, t, :]  # Add temporal info        │
│                                                                              │
│ Stack all timesteps:                                                         │
│   num_tokens_list = [tokens_0, tokens_1, ..., tokens_9]                    │
│   num_tokens = cat(num_tokens_list, dim=1)                                 │
│   →  [32, 80, 128]  # [batch, seq_len * num_numerical, d_model]           │
│                      # 10 timesteps × 8 features = 80 tokens               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: CATEGORICAL TOKENIZATION (STATIC FEATURES)                          │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ Logarithmic Embedding Dimension Scaling:                                    │
│   FORMULA: emb_dim = int(8 * log2(cardinality + 1))                        │
│            emb_dim = clamp(emb_dim, d_model/4, d_model)                    │
│                                                                              │
│ For our example:                                                             │
│   symbol: cardinality=100 → emb_dim = int(8*log2(101)) ≈ 53               │
│   sector: cardinality=5   → emb_dim = int(8*log2(6)) ≈ 20 → clamped to 32 │
│                                                                              │
│ Symbol Processing:                                                           │
│   symbol_indices = x_cat[:, 0]  # [32]                                     │
│   symbol_emb = Embedding(100, 53)(symbol_indices)  # [32, 53]              │
│   symbol_proj = Linear(53, 128)(symbol_emb)        # [32, 128]             │
│                                                                              │
│ Sector Processing:                                                           │
│   sector_indices = x_cat[:, 1]  # [32]                                     │
│   sector_emb = Embedding(5, 32)(sector_indices)    # [32, 32]              │
│   sector_proj = Linear(32, 128)(sector_emb)        # [32, 128]             │
│                                                                              │
│ Stack categorical tokens:                                                    │
│   cat_tokens_list = [symbol_proj, sector_proj]                             │
│   cat_tokens = stack(cat_tokens_list, dim=1)                               │
│   →  [32, 2, 128]  # [batch, num_categorical, d_model]                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: TOKEN ASSEMBLY WITH CLS                                             │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ Token Ordering: [CLS, numerical_tokens, categorical_tokens]                 │
│                                                                              │
│ 1. CLS Token (learnable aggregation token):                                 │
│      cls_token = CLSToken(d_model)              # [1, 1, 128]              │
│      cls_expanded = cls_token.expand(32, -1, -1) # [32, 1, 128]            │
│                                                                              │
│ 2. Numerical Tokens (time-varying):                                         │
│      num_tokens: [32, 80, 128]  # 10 timesteps × 8 features                │
│                                                                              │
│ 3. Categorical Tokens (static):                                             │
│      cat_tokens: [32, 2, 128]  # 2 categorical features                    │
│                                                                              │
│ Concatenate all tokens:                                                     │
│   all_tokens = cat([cls_expanded, num_tokens, cat_tokens], dim=1)         │
│   →  [32, 83, 128]  # [batch, 1 + 80 + 2, d_model]                        │
│                      # 1 CLS + 80 numerical + 2 categorical = 83 tokens    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: UNIFIED TRANSFORMER PROCESSING                                      │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ Multi-Head Self-Attention (3 layers, num_heads=4):                          │
│                                                                              │
│ Layer configuration:                                                         │
│   d_model = 128                                                              │
│   num_heads = 4                                                              │
│   d_head = 128 / 4 = 32                                                      │
│   d_ffn = 4 * 128 = 512                                                      │
│                                                                              │
│ For each transformer layer:                                                  │
│                                                                              │
│   A. Multi-Head Self-Attention:                                              │
│      Q = all_tokens @ W_q  # [32, 83, 128] @ [128, 128] → [32, 83, 128]   │
│      K = all_tokens @ W_k  # [32, 83, 128] @ [128, 128] → [32, 83, 128]   │
│      V = all_tokens @ W_v  # [32, 83, 128] @ [128, 128] → [32, 83, 128]   │
│                                                                              │
│      Reshape for multi-head attention:                                       │
│      Q = Q.view(32, 83, 4, 32).transpose(1, 2)  # [32, 4, 83, 32]          │
│      K = K.view(32, 83, 4, 32).transpose(1, 2)  # [32, 4, 83, 32]          │
│      V = V.view(32, 83, 4, 32).transpose(1, 2)  # [32, 4, 83, 32]          │
│                                                                              │
│      Attention computation:                                                  │
│      scores = Q @ K^T  # [32, 4, 83, 32] @ [32, 4, 32, 83] → [32, 4, 83, 83]│
│      attn_weights = softmax(scores / √32)  # [32, 4, 83, 83]               │
│      attn_output = attn_weights @ V  # [32, 4, 83, 83] @ [32, 4, 83, 32]  │
│                                       # → [32, 4, 83, 32]                   │
│                                                                              │
│      Concatenate heads:                                                      │
│      attn_output = attn_output.transpose(1, 2).reshape(32, 83, 128)        │
│      output = LayerNorm(all_tokens + attn_output)  # [32, 83, 128]         │
│                                                                              │
│   B. Feed-Forward Network:                                                   │
│      ffn_hidden = Linear(128, 512)(output)     # [32, 83, 512]             │
│      ffn_hidden = ReLU(ffn_hidden)             # [32, 83, 512]             │
│      ffn_output = Linear(512, 128)(ffn_hidden) # [32, 83, 128]             │
│      output = LayerNorm(output + ffn_output)   # [32, 83, 128]             │
│                                                                              │
│ Final transformer output: [32, 83, 128]                                      │
│                                                                              │
│ KEY INSIGHT: All tokens (CLS, numerical at each timestep, categorical)      │
│              attend to each other, enabling rich cross-feature interactions │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: CLS EXTRACTION & PREDICTION                                         │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ Extract CLS token (aggregated representation):                              │
│   cls_output = transformer_output[:, 0, :]  # [32, 128]                    │
│                                                                              │
│ Prediction Head (MultiHorizonHead):                                         │
│   predictions = Linear(128, output_dim)(cls_output)                         │
│   →  [32, 1]  # [batch, output_dim]                                        │
│                                                                              │
│ For multi-horizon forecasting (output_dim=3):                               │
│   predictions = Linear(128, 3)(cls_output)                                  │
│   →  [32, 3]  # Forecasts for t+1, t+2, t+3                                │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 KEY ARCHITECTURAL FEATURES:

1. Unified Tokenization:
   - All features converted to same d_model dimensional space
   - Numerical: Linear transformation per feature
   - Categorical: Embedding + projection to d_model

2. Temporal Positional Encoding:
   - Added to numerical tokens to preserve time ordering
   - Learnable: pos_emb[t] ∈ ℝ^{d_model} for each timestep t

3. Single Shared Transformer:
   - All tokens processed together in unified attention
   - CLS token attends to all numerical and categorical tokens
   - Enables cross-modality feature interactions

4. Logarithmic Embedding Scaling:
   - emb_dim = int(8 * log2(cardinality + 1))
   - Bounds: [d_model/4, d_model]
   - Matches embedding capacity to information content

🧠 COMPUTATIONAL COMPLEXITY:

Memory Usage (per forward pass):
- All tokens: 32 × 83 × 128 × 4 bytes ≈ 1.36 MB
- Attention matrix: 32 × 4 × 83² × 4 bytes ≈ 3.5 MB (per layer)
- Total: ~1.36 MB + ~10.5 MB (3 layers) ≈ 11.86 MB

Time Complexity:
- Tokenization: O(sequence_length × num_numerical × d_model)
- Self-Attention: O(num_layers × num_tokens² × d_model)
  where num_tokens = 1 + (sequence_length × num_numerical) + num_categorical
  = 1 + (10 × 8) + 2 = 83
- FFN: O(num_layers × num_tokens × d_model × d_ffn)
- Dominated by attention: O(L × T² × d) where T=83

🎨 ADVANTAGES OVER DUAL-PATH ARCHITECTURES:

1. Richer Feature Interactions:
   - Categorical features can directly attend to specific timesteps
   - Example: "sector" token can learn which historical prices are most relevant
   - Numerical tokens can attend to categorical context

2. Simpler Architecture:
   - Single transformer instead of two separate paths
   - Fewer parameters and simpler training dynamics
   - No fusion layer needed

3. Flexible Attention Patterns:
   - Model learns optimal attention between all feature types
   - No artificial separation between categorical and numerical
   - Better for features with complex interdependencies

4. Graceful Feature Handling:
   - Works seamlessly when num_categorical=0 (numerical only)
   - Simply omits categorical tokens from sequence
   - No architectural changes needed

⚡ USAGE NOTES:

1. When num_categorical = 0:
   - Only numerical tokens and CLS are used
   - Token sequence: [CLS, num_tokens]
   - Total tokens: 1 + (sequence_length × num_numerical) = 81

2. Feature Preparation:
   - Numerical: normalized/scaled continuous values
   - Categorical: integer indices [0, cardinality-1]
   - Ensure cardinalities match actual unique values

3. Training Considerations:
   - Use learning rate warmup for stable training
   - Gradient clipping helps with long sequences
   - Dropout applied uniformly to all tokens

4. Memory vs. Dual-Path:
   - Unified: O(T²) where T = all tokens = 83
   - Dual: O(T_cat²) + O(T_num²) where T_cat=3, T_num=11
   - Unified: 83² = 6,889 vs Dual: 3² + 11² = 9 + 121 = 130
   - Unified has higher memory cost but richer interactions

This unified architecture is optimal for scenarios where cross-modal feature interactions
are important, such as learning how specific stock symbols respond differently to the
same market conditions, or how sector membership influences price pattern interpretation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import numpy as np

# Import shared base components
from .base.embeddings import CLSToken, NumericalTokenizer, CategoricalEmbedding
from .base.prediction_heads import MultiHorizonHead
from .base.model_interface import TransformerBasedModel


class FeatureTokenizer(nn.Module):
    """Converts tabular features into transformer tokens."""
    
    def __init__(self, num_numerical: int, cat_cardinalities: List[int], d_token: int):
        super().__init__()
        
        # Validation
        if num_numerical + len(cat_cardinalities) == 0:
            logging.error("Must have at least one feature")
            raise ValueError("Must have at least one feature")
        
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        
        # CLS token for prediction (using shared base component)
        # Learnable parameter that aggregates all feature information
        self.cls_token = CLSToken(d_token)

        # Numerical feature tokenization using shared base component
        # Each numerical feature gets its own weight vector and bias vector
        if num_numerical > 0:
            self.num_tokenizer = NumericalTokenizer(num_numerical, d_token)

        # Categorical feature embeddings using shared base component
        # Each categorical feature gets its own embedding table: vocab_size → d_token
        if cat_cardinalities:
            self.cat_embedding = CategoricalEmbedding(cat_cardinalities, d_token)
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Convert raw features into tokens for transformer processing.

        Args:
            x_num: Numerical features [batch_size, num_numerical]
            x_cat: Categorical features [batch_size, num_categorical]

        Returns:
            tokens: [batch_size, 1 + num_features, d_token]
                   First token is CLS, followed by feature tokens
        """
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        tokens = []

        # Process numerical features using shared base tokenizer
        if self.num_numerical > 0 and x_num is not None:
            num_tokens = self.num_tokenizer(x_num)  # [batch, num_features, d_token]
            tokens.append(num_tokens)

        # Process categorical features using shared base embedding
        if self.cat_cardinalities and x_cat is not None:
            cat_tokens = self.cat_embedding(x_cat)  # [batch, num_categorical, d_token]
            tokens.append(cat_tokens)

        # Combine all feature tokens along feature dimension
        if not tokens:
            raise ValueError("No valid input features provided")

        feature_tokens = torch.cat(tokens, dim=1)  # [batch, total_features, d_token]

        # Prepend CLS token for final prediction aggregation (using shared base component)
        # CLS token will collect information from all features via self-attention
        cls_tokens = self.cls_token.expand(batch_size)  # [batch, 1, d_token]
        return torch.cat([cls_tokens, feature_tokens], dim=1)    # [batch, 1+features, d_token]


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
        
        # CLS token for prediction (using shared base component)
        self.cls_token = CLSToken(d_token)

        # Numerical feature tokenization using shared base component (same for all timesteps)
        if num_numerical > 0:
            self.num_tokenizer = NumericalTokenizer(num_numerical, d_token)

        # Categorical feature embeddings using shared base component (same for all timesteps)
        if cat_cardinalities:
            self.cat_embedding = CategoricalEmbedding(cat_cardinalities, d_token)
        
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
            
            # Process numerical features for this timestep (using shared base tokenizer)
            if self.num_numerical > 0 and x_num_seq is not None:
                x_num_t = x_num_seq[:, t, :]  # [batch, num_numerical]
                num_tokens = self.num_tokenizer(x_num_t)  # [batch, num_numerical, d_token]
                timestep_tokens.append(num_tokens)

            # Process categorical features for this timestep (using shared base embedding)
            if self.cat_cardinalities and x_cat_seq is not None:
                x_cat_t = x_cat_seq[:, t, :].long()  # [batch, num_categorical]
                cat_tokens = self.cat_embedding(x_cat_t)  # [batch, num_categorical, d_token]
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
        
        # Prepend CLS token (using shared base component)
        cls_tokens = self.cls_token.expand(batch_size)
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
        
        # Transformer encoder - stack of identical layers with self-attention + FFN
        # Each layer: MultiHeadAttention → Add&Norm → FeedForward → Add&Norm
        # norm_first=True uses pre-normalization for better gradient flow
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,          # Token dimension (input/output size)
            nhead=n_heads,            # Number of attention heads
            dim_feedforward=d_ffn,    # Hidden size in feed-forward network
            dropout=dropout,          # Dropout rate
            activation=activation,    # Activation function (GELU/ReLU)
            batch_first=True,         # Input format: [batch, seq, features]
            norm_first=True           # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output dimension for downstream prediction head
        self.d_token = d_token
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_num: Numerical features [batch_size, num_numerical]
            x_cat: Categorical features [batch_size, num_categorical]
            
        Returns:
            cls_output: [batch_size, d_token] - CLS token representation
        """
        # Step 1: Convert features to tokens
        # tokens: [batch_size, 1 + num_features, d_token]
        tokens = self.tokenizer(x_num, x_cat)

        # Step 2: Apply self-attention across all tokens
        # Each token can attend to every other token, learning feature interactions
        # transformer_out: [batch_size, 1 + num_features, d_token]
        transformer_out = self.transformer(tokens)

        # Step 3: Extract CLS token for prediction
        # CLS token (first token) has aggregated information from all features
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
        prediction_horizons: int = 1,
        **transformer_kwargs
    ):
        super().__init__()

        self.ft_transformer = FTTransformer(num_numerical, cat_cardinalities, **transformer_kwargs)
        self.prediction_horizons = prediction_horizons

        # Prediction head using shared base component
        d_token = self.ft_transformer.d_token
        self.head = MultiHorizonHead(
            d_input=d_token,
            prediction_horizons=prediction_horizons,
            hidden_dim=None,  # Simple linear head
            dropout=transformer_kwargs.get('dropout', 0.1)
        )

        self.n_classes = n_classes
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_num: Numerical features
            x_cat: Categorical features

        Returns:
            predictions: [batch_size, n_classes * prediction_horizons]
                       For regression: [batch_size, prediction_horizons]
        """
        # Get CLS token representation from transformer
        cls_output = self.ft_transformer(x_num, x_cat)  # [batch, d_token]

        # Apply prediction head (MultiHorizonHead handles reshaping automatically)
        return self.head(cls_output)


class SequenceFTTransformerPredictor(nn.Module):
    """Complete Sequential FT-Transformer with prediction head."""

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        sequence_length: int,
        n_classes: int = 1,
        prediction_horizons: int = 1,
        **transformer_kwargs
    ):
        super().__init__()

        self.sequence_ft_transformer = SequenceFTTransformer(
            num_numerical, cat_cardinalities, sequence_length, **transformer_kwargs
        )
        self.prediction_horizons = prediction_horizons

        # Prediction head using shared base component
        d_token = self.sequence_ft_transformer.d_token
        self.head = MultiHorizonHead(
            d_input=d_token,
            prediction_horizons=prediction_horizons,
            hidden_dim=None,  # Simple linear head
            dropout=transformer_kwargs.get('dropout', 0.1)
        )

        self.n_classes = n_classes
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Sequential features [batch_size, sequence_length, num_features]

        Returns:
            predictions: [batch_size, n_classes * prediction_horizons]
                       For regression: [batch_size, prediction_horizons]
        """
        cls_output = self.sequence_ft_transformer(x_seq)
        # MultiHorizonHead handles reshaping automatically
        return self.head(cls_output)


class FTTransformerCLSModel(TransformerBasedModel):
    """
    🧠 FT-TRANSFORMER WITH CLS TOKEN FOR CATEGORICAL FEATURES (FT_TRANSFORMER_CLS)
    ==============================================================================

    This model extends the FT-Transformer architecture to handle STATIC categorical features
    alongside TIME-VARYING numerical features for time series forecasting.

    📊 ARCHITECTURE OVERVIEW:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Numerical Sequences (3D) + Categorical Features (2D) → Unified Tokens →    │
    │  Transformer → CLS Token → Prediction                                       │
    └─────────────────────────────────────────────────────────────────────────────┘

    🔄 DATA FLOW WITH MATRIX DIMENSIONS:

    Example Configuration:
    - batch_size = 32
    - sequence_length = 10
    - num_numerical = 8 (price, volume, seasonal features, etc.)
    - num_categorical = 2 (symbol, sector)
    - cat_cardinalities = [100, 5] (100 symbols, 5 sectors)
    - d_model = 128 (embedding dimension)

    Step 1: Input Format
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ x_num: [32, 10, 8]     # Numerical sequences (time-varying)                 │
    │                        # Shape: [batch, seq_len, num_numerical_features]    │
    │                                                                              │
    │ x_cat: [32, 2]         # Categorical features (static per sequence)         │
    │                        # Shape: [batch, num_categorical_features]           │
    │                        # Values are integer indices (label encoded)         │
    └─────────────────────────────────────────────────────────────────────────────┘

    Step 2: Categorical Embedding with Logarithmic Dimension Scaling
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ EMBEDDING DIMENSION FORMULA:                                                │
    │   emb_dim = int(8 * log2(cardinality + 1))                                 │
    │   emb_dim = clamp(emb_dim, d_model/4, d_model)                             │
    │                                                                              │
    │ RATIONALE:                                                                   │
    │   - Logarithmic scaling: Information-theoretic capacity                     │
    │   - Lower bound (d_model/4): Ensures minimum representational capacity     │
    │   - Upper bound (d_model): Prevents excessive dimensions                   │
    │                                                                              │
    │ EXAMPLES (d_model=128):                                                      │
    │   cardinality=10   → emb_dim = int(8*log2(11)) = 27 → clamp → 32          │
    │   cardinality=100  → emb_dim = int(8*log2(101)) = 53 → clamp → 53         │
    │   cardinality=1000 → emb_dim = int(8*log2(1001)) = 79 → clamp → 79        │
    │   cardinality=10000 → emb_dim = int(8*log2(10001)) = 106 → clamp → 106    │
    │                                                                              │
    │ CATEGORICAL EMBEDDING:                                                       │
    │   symbol_embedding: Embedding(100, 53)   # 100 symbols → 53 dims           │
    │   sector_embedding: Embedding(5, 32)     # 5 sectors → 32 dims             │
    │                                                                              │
    │   x_cat: [32, 2] (integer indices)                                          │
    │   → symbol_emb: [32, 53]                                                    │
    │   → sector_emb: [32, 32]                                                    │
    │                                                                              │
    │ PROJECT TO d_model:                                                          │
    │   symbol_proj: Linear(53, 128) → [32, 128]                                 │
    │   sector_proj: Linear(32, 128) → [32, 128]                                 │
    │   cat_tokens: [32, 2, 128]  # Stacked categorical tokens                   │
    └─────────────────────────────────────────────────────────────────────────────┘

    Step 3: Numerical Tokenization
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ For each numerical feature at each timestep:                                │
    │   token_ij = x_ij * W_j + b_j                                              │
    │                                                                              │
    │ num_tokenizer: NumericalTokenizer(8, 128)                                   │
    │                                                                              │
    │ Process each timestep:                                                       │
    │   for t in range(10):                                                        │
    │     x_num_t = x_num[:, t, :]  # [32, 8]                                    │
    │     tokens_t = num_tokenizer(x_num_t)  # [32, 8, 128]                      │
    │                                                                              │
    │ Concatenate all timesteps:                                                  │
    │   num_tokens: [32, 80, 128]  # 10 timesteps × 8 features                   │
    └─────────────────────────────────────────────────────────────────────────────┘

    Step 4: Token Assembly (UNIFIED PROCESSING)
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ Token Order: [CLS, numerical_t0...t9, categorical]                          │
    │                                                                              │
    │ 1. CLS Token: [32, 1, 128]  # Learnable aggregation token                  │
    │ 2. Numerical Tokens: [32, 80, 128]  # 10 timesteps × 8 features            │
    │ 3. Categorical Tokens: [32, 2, 128]  # 2 categorical features              │
    │                                                                              │
    │ Concatenate along sequence dimension:                                       │
    │   all_tokens = cat([CLS, num_tokens, cat_tokens], dim=1)                   │
    │   → [32, 83, 128]  # 1 CLS + 80 numerical + 2 categorical                 │
    └─────────────────────────────────────────────────────────────────────────────┘

    Step 5: Transformer Processing
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ Multi-Head Self-Attention (num_heads=8):                                    │
    │   d_head = 128 / 8 = 16                                                     │
    │                                                                              │
    │   Q, K, V = all_tokens @ W_q, W_k, W_v  # [32, 83, 128]                   │
    │   Reshape: [32, 8, 83, 16]  # [batch, heads, seq, d_head]                 │
    │                                                                              │
    │   Attention(Q,K,V) = softmax(QK^T/√16)V                                     │
    │   scores: [32, 8, 83, 83]  # All tokens attend to all tokens              │
    │   output: [32, 83, 128]                                                     │
    │                                                                              │
    │ Feed-Forward Network:                                                        │
    │   FFN(x) = Linear(GELU(Linear(x, 512)), 128)                               │
    │   output: [32, 83, 128]                                                     │
    │                                                                              │
    │ Repeat for num_layers (typically 3-6)                                       │
    └─────────────────────────────────────────────────────────────────────────────┘

    Step 6: Prediction from CLS Token
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ Extract CLS token (first token):                                            │
    │   cls_output = transformer_output[:, 0, :]  # [32, 128]                    │
    │                                                                              │
    │ Prediction Head:                                                             │
    │   predictions = Linear(cls_output, output_dim)  # [32, output_dim]         │
    │                                                                              │
    │ For multi-horizon forecasting (prediction_horizon=3):                       │
    │   output_dim = 3                                                             │
    │   predictions: [32, 3]  # Predictions for horizons 1, 2, 3                 │
    └─────────────────────────────────────────────────────────────────────────────┘

    🎯 KEY DESIGN DECISIONS:

    1. **Categorical Embedding Dimensions**: Logarithmic scaling with bounds
       - Formula: emb_dim = clamp(int(8 * log2(cardinality + 1)), d_model/4, d_model)
       - Balances capacity with efficiency
       - Information-theoretic foundation

    2. **Token Order**: CLS → Numerical (time-ordered) → Categorical
       - CLS token aggregates all information
       - Numerical tokens preserve temporal structure
       - Categorical tokens provide static context

    3. **Unified Processing**: Single transformer processes all tokens
       - Attention mechanism learns optimal feature interactions
       - Categorical features attend to all timesteps
       - Numerical features attend to categorical context

    4. **Static vs Dynamic Features**:
       - Numerical: 3D sequences (time-varying)
       - Categorical: 2D features (static per sequence)
       - Categorical extracted from LAST timestep of each sequence

    🚀 ADVANTAGES:

    1. **Flexible Feature Handling**: Handles heterogeneous feature types
    2. **Temporal + Static Context**: Combines time-varying and static information
    3. **Automatic Interaction Discovery**: Attention learns complex relationships
    4. **Scalable**: Logarithmic embedding dimensions scale efficiently

    ⚠️ IMPORTANT NOTES:

    1. **Unseen Categories**: Will throw an error (by design)
       - Categorical features must be label encoded during training
       - New categories at inference time are not supported

    2. **Input Format**:
       - x_num: Numerical sequences [batch, seq_len, num_numerical]
       - x_cat: Categorical indices [batch, num_categorical] (dtype: long)

    3. **Categorical Columns**: Must be specified and encoded in predictor.py
       - Encoding: sklearn.preprocessing.LabelEncoder
       - Sequence creation: Extract from last timestep

    This implementation provides a robust foundation for time series forecasting with
    mixed feature types, combining temporal dynamics with static categorical context.
    """

    def __init__(
        self,
        sequence_length: int,
        num_numerical: int,
        num_categorical: int,
        cat_cardinalities: List[int],
        output_dim: int,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize FT-Transformer with CLS token for categorical features.

        Args:
            sequence_length: Length of input sequences (lookback window)
            num_numerical: Number of numerical features per timestep
            num_categorical: Number of categorical features
            cat_cardinalities: List of cardinalities for each categorical feature
            output_dim: Output dimension (num_targets * prediction_horizon)
            d_token: Embedding dimension (token size)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__(d_token=d_token, n_heads=n_heads, n_layers=n_layers)

        self.sequence_length = sequence_length
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.cat_cardinalities = cat_cardinalities
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.activation_name = activation

        # Validate inputs
        if num_categorical > 0 and len(cat_cardinalities) != num_categorical:
            raise ValueError(f"cat_cardinalities length ({len(cat_cardinalities)}) must match num_categorical ({num_categorical})")

        # CLS token for prediction
        self.cls_token = CLSToken(d_token)

        # Numerical tokenizer
        if num_numerical > 0:
            self.num_tokenizer = NumericalTokenizer(num_numerical, d_token)

        # Categorical embeddings with logarithmic dimension scaling
        if num_categorical > 0:
            import math

            self.cat_embeddings = nn.ModuleList()
            self.cat_projections = nn.ModuleList()

            for cardinality in cat_cardinalities:
                # Calculate embedding dimension using logarithmic scaling
                emb_dim = int(8 * math.log2(cardinality + 1))
                # Clamp to bounds [d_token/4, d_token]
                min_dim = d_token // 4
                max_dim = d_token
                emb_dim = max(min_dim, min(max_dim, emb_dim))

                # Create embedding layer
                embedding = nn.Embedding(cardinality, emb_dim)
                self.cat_embeddings.append(embedding)

                # Project to d_token if needed
                if emb_dim != d_token:
                    projection = nn.Linear(emb_dim, d_token)
                    self.cat_projections.append(projection)
                else:
                    self.cat_projections.append(nn.Identity())

        # Positional encoding for temporal sequences (optional, can help)
        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, sequence_length, d_token) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head
        self.head = MultiHorizonHead(
            d_input=d_token,
            prediction_horizons=output_dim,
            hidden_dim=None,
            dropout=dropout
        )

    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with separate numerical and categorical inputs.

        Args:
            x_num: Numerical sequences [batch_size, sequence_length, num_numerical]
            x_cat: Categorical features [batch_size, num_categorical] (integer indices)

        Returns:
            predictions: [batch_size, output_dim]
        """
        batch_size = x_num.shape[0]

        # Step 1: Create CLS token
        cls_tokens = self.cls_token(batch_size)  # [batch, 1, d_token]

        # Step 2: Tokenize numerical sequences
        # Process each timestep independently
        num_tokens_list = []
        for t in range(self.sequence_length):
            x_num_t = x_num[:, t, :]  # [batch, num_numerical]
            tokens_t = self.num_tokenizer(x_num_t)  # [batch, num_numerical, d_token]
            # Add temporal positional encoding
            tokens_t = tokens_t + self.temporal_pos_encoding[:, t:t+1, :].expand(-1, self.num_numerical, -1)
            num_tokens_list.append(tokens_t)

        # Concatenate all timesteps: [batch, seq_len * num_numerical, d_token]
        num_tokens = torch.cat(num_tokens_list, dim=1)

        # Step 3: Process categorical features (if present)
        if x_cat is not None and self.num_categorical > 0:
            cat_tokens_list = []
            for i in range(self.num_categorical):
                cat_indices = x_cat[:, i]  # [batch]
                emb = self.cat_embeddings[i](cat_indices)  # [batch, emb_dim]
                projected = self.cat_projections[i](emb)  # [batch, d_token]
                cat_tokens_list.append(projected.unsqueeze(1))  # [batch, 1, d_token]

            cat_tokens = torch.cat(cat_tokens_list, dim=1)  # [batch, num_categorical, d_token]

            # Step 4: Concatenate all tokens: [CLS, numerical, categorical]
            all_tokens = torch.cat([cls_tokens, num_tokens, cat_tokens], dim=1)
        else:
            # No categorical features
            all_tokens = torch.cat([cls_tokens, num_tokens], dim=1)

        # Step 5: Transformer processing
        transformer_output = self.transformer(all_tokens)  # [batch, num_tokens, d_token]

        # Step 6: Extract CLS token and predict
        cls_output = transformer_output[:, 0, :]  # [batch, d_token]
        predictions = self.head(cls_output)  # [batch, output_dim]

        return predictions

    def get_model_config(self) -> Dict[str, Any]:
        """Get the current configuration of the model."""
        return {
            'model_type': 'ft_transformer_cls',
            'd_token': self.d_token,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'dropout': self.dropout_rate,
            'activation': self.activation_name,
            'sequence_length': self.sequence_length,
            'num_numerical': self.num_numerical,
            'num_categorical': self.num_categorical,
            'cat_cardinalities': self.cat_cardinalities,
            'output_dim': self.output_dim
        }

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from the last forward pass.

        Returns:
            Attention weights tensor or None if not available.
        """
        return None  # Not implemented for CLS models
