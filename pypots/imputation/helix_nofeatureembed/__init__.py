"""
The package of the partially-observed time-series imputation model HELIX_NoFeatureEmbed.

Ablation: Remove learnable feature identity embedding
"""

# Created by MiBah Cat <milaogou@gmail.com>
# License: BSD-3-Clause

from .model import HELIX_NoFeatureEmbed

__all__ = [
    "HELIX_NoFeatureEmbed",
]
