"""
The core wrapper assembles the submodules of HELIX_NoHybrid imputation model.
ABLATION: Remove parallel encoding, use only serial cross-encoding
"""

# Created by MiBah Cat <milaogou@gmail.com>
# License: BSD-3-Clause

import math
import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding for temporal dimension."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for rotary positional encoding"
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, positions):
        return self.pe[positions]


class TimeSeriesEmbedding2D(nn.Module):
    """Embedding layer for 2D time series data [B, T, F]."""
    
    def __init__(self, n_features, pe_dim=16, feature_embed_dim=1):
        super().__init__()
        self.n_features = n_features
        self.pe_dim = pe_dim
        self.feature_embed_dim = feature_embed_dim
        
        self.temporal_pe = RotaryPositionalEncoding(d_model=pe_dim)
        self.feature_id = nn.Parameter(torch.randn(n_features, feature_embed_dim))
    
    def forward(self, X, missing_mask):
        B, T, F = X.shape
        device = X.device
        
        data_val = X.unsqueeze(-1)
        
        pos_indices = torch.arange(T, device=device)
        temporal_encoding = self.temporal_pe(pos_indices)
        temporal_encoding = temporal_encoding.unsqueeze(0).unsqueeze(2)
        temporal_encoding = temporal_encoding.expand(B, T, F, self.pe_dim)
        
        feature_embedding = self.feature_id.unsqueeze(0).unsqueeze(0)
        feature_embedding = feature_embedding.expand(B, T, F, self.feature_embed_dim)
        
        mask_feature = missing_mask.unsqueeze(-1)
        
        embedded = torch.cat([
            data_val,
            temporal_encoding,
            feature_embedding,
            mask_feature
        ], dim=-1)
        
        return embedded


class FeatureProjection(nn.Module):
    """Project between embedding dimension and model dimension."""
    
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.forward_proj = nn.Linear(input_dim, d_model)
        self.backward_proj = nn.Linear(d_model, input_dim)
        
    def project_forward(self, x):
        return self.forward_proj(x)
        
    def project_backward(self, x):
        return self.backward_proj(x)


class UnifiedAttentionEncoder(nn.Module):
    """Unified attention encoder that can be applied to any dimension."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x


class DimensionalAttention(nn.Module):
    """Apply attention along a specific dimension."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.encoder = UnifiedAttentionEncoder(d_model, n_heads, dropout)
    
    def forward(self, x, target_dim):
        B, T, F, D = x.shape
        
        if target_dim == 'time':
            x_reshaped = x.permute(0, 2, 1, 3).reshape(B * F, T, D)
            out = self.encoder(x_reshaped)
            out = out.reshape(B, F, T, D).permute(0, 2, 1, 3)
            
        elif target_dim == 'feature':
            x_reshaped = x.reshape(B * T, F, D)
            out = self.encoder(x_reshaped)
            out = out.reshape(B, T, F, D)
        else:
            raise ValueError(f"Invalid target_dim: {target_dim}")
            
        return out


class BackboneHELIX_NoHybrid(nn.Module):
    """
    HELIX backbone with SERIAL-ONLY encoding (ABLATION).
    Only uses: Time → Feature → Time (no parallel encoding)
    """
    
    def __init__(self, n_features, pe_dim, feature_embed_dim, d_model, n_heads, n_layers, dropout):
        super().__init__()
        
        embed_dim = pe_dim + feature_embed_dim + 2
        self.embedding = TimeSeriesEmbedding2D(n_features, pe_dim, feature_embed_dim)
        
        self.projection = FeatureProjection(embed_dim, d_model)
        
        # Multi-layer encoders (same structure, different usage)
        self.encoders = nn.ModuleList([
            nn.ModuleDict({
                'time': DimensionalAttention(d_model, n_heads, dropout),
                'feature': DimensionalAttention(d_model, n_heads, dropout),
            })
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(self, X, missing_mask):
        """
        ABLATION: Only serial cross-encoding (Time → Feature → Time)
        No parallel encoding
        """
        # Embedding
        embedded = self.embedding(X, missing_mask)
        
        # Project to model dimension
        x = self.projection.project_forward(embedded)
        
        # Store all intermediate outputs for multi-level fusion
        all_outputs = [x]
        
        # Multi-layer SERIAL encoding (ABLATION)
        for layer_encoders in self.encoders:
            # ABLATION: Remove Phase 1 parallel encoding
            # Only use serial cross-encoding: Time → Feature → Time
            
            # Step 1: Time attention
            x = layer_encoders['time'](x, 'time')
            all_outputs.append(x)
            
            # Step 2: Feature attention
            x = layer_encoders['feature'](x, 'feature')
            all_outputs.append(x)
            
            # Step 3: Time attention again (complete the cycle)
            x = layer_encoders['time'](x, 'time')
            all_outputs.append(x)
        
        # Global fusion
        fused = torch.stack(all_outputs, dim=0).mean(dim=0)
        fused = self.final_norm(fused)
        
        # Project back to 1D
        output = self.output_proj(fused).squeeze(-1)
        
        return output


class _HELIX_NoHybrid(ModelCore):
    """Core model wrapper for HELIX_NoHybrid."""
    
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        pe_dim: int,
        feature_embed_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        
        self.n_steps = n_steps
        self.n_features = n_features
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.training_loss = training_loss
        
        if validation_metric.__class__.__name__ == "Criterion":
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric
        
        self.backbone = BackboneHELIX_NoHybrid(
            n_features=n_features,
            pe_dim=pe_dim,
            feature_embed_dim=feature_embed_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
    
    def forward(self, inputs: dict, calc_criterion: bool = False) -> dict:
        X = inputs["X"]
        missing_mask = inputs["missing_mask"]
        
        # Forward pass
        reconstruction = self.backbone(X, missing_mask)
        
        # Replace observed values with original data
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }
        
        if calc_criterion:
            X_ori = inputs["X_ori"]
            indicating_mask = inputs["indicating_mask"]
            
            if self.training:
                ORT_loss = self.ORT_weight * self.training_loss(reconstruction, X, missing_mask)
                MIT_loss = self.MIT_weight * self.training_loss(reconstruction, X_ori, indicating_mask)
                loss = ORT_loss + MIT_loss
                
                results["ORT_loss"] = ORT_loss
                results["MIT_loss"] = MIT_loss
                results["loss"] = loss
            else:
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)
        
        return results