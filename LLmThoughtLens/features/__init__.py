"""Features layer — Feature, FeatureSet, SAE (+ pretrained loaders), FeatureExtractor, ActivationCache, FeatureLabeler, FeatureIntervention."""

from LLmThoughtLens.features.cache import ActivationCache
from LLmThoughtLens.features.extractor import (
    FeatureExtractor,
    SAEAttachment,
    SAEHookUnavailableError,
    sae_input_activations,
)
from LLmThoughtLens.features.feature import Feature, FeatureSet
from LLmThoughtLens.features.intervention import FeatureIntervention, InterventionMode
from LLmThoughtLens.features.labeler import FeatureLabeler
from LLmThoughtLens.features.sae import (
    SAE_ARCHITECTURES,
    SAE_SITES,
    SAEConfig,
    SparseAutoencoder,
    hook_name_for,
    parse_hook_name,
)
from LLmThoughtLens.features.sae_loaders import (
    PRETRAINED_RELEASES,
    describe_pretrained,
    from_pretrained,
    list_pretrained,
    load_gemma_scope,
    load_saelens,
)

__all__ = [
    "Feature",
    "FeatureSet",
    "FeatureExtractor",
    "FeatureIntervention",
    "InterventionMode",
    "SparseAutoencoder",
    "SAEConfig",
    "SAE_ARCHITECTURES",
    "SAE_SITES",
    "SAEAttachment",
    "SAEHookUnavailableError",
    "parse_hook_name",
    "hook_name_for",
    "sae_input_activations",
    "load_saelens",
    "load_gemma_scope",
    "from_pretrained",
    "describe_pretrained",
    "list_pretrained",
    "PRETRAINED_RELEASES",
    "ActivationCache",
    "FeatureLabeler",
]
