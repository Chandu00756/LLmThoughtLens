"""llmscope.features — sparse feature representations and SAE utilities."""

from llmscope.features.feature import Feature, FeatureSet
from llmscope.features.extractor import FeatureExtractor
from llmscope.features.intervention import FeatureIntervention
from llmscope.features.sae import SparseAutoencoder, SAEConfig

__all__ = [
    "Feature",
    "FeatureSet",
    "FeatureExtractor",
    "FeatureIntervention",
    "SparseAutoencoder",
    "SAEConfig",
]
