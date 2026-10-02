"""Shared white-box model layer: :class:`HookedModel` + architecture-family resolution.

Everything that touches a local HuggingFace model's internals (the
HuggingFace provider, the live X-ray, interventions, gradient attribution,
SAE hook points, steering) goes through this package so there is exactly one
residual-stream convention and one hook mechanism.  See
:mod:`LLmThoughtLens.models.hooked` for the convention.

Importing this package never imports torch / transformers.
"""

from __future__ import annotations

from LLmThoughtLens.models.families import (
    FAMILIES,
    Architecture,
    FamilySpec,
    UnsupportedArchitectureError,
    family_for_model_type,
    resolve_architecture,
    resolve_family,
    resolve_final_norm,
    resolve_transformer_blocks,
    supported_families,
)
from LLmThoughtLens.models.hooked import (
    SITES,
    ForwardResult,
    GenerateResult,
    HookedModel,
    HookSite,
    ResidHook,
    TokenizedPrompt,
    infer_device,
    load_hf_model,
    register_site_hook,
    resolve_device,
    resolve_dtype,
)

__all__ = [
    "FAMILIES",
    "SITES",
    "Architecture",
    "FamilySpec",
    "ForwardResult",
    "GenerateResult",
    "HookSite",
    "HookedModel",
    "ResidHook",
    "TokenizedPrompt",
    "UnsupportedArchitectureError",
    "family_for_model_type",
    "infer_device",
    "load_hf_model",
    "register_site_hook",
    "resolve_architecture",
    "resolve_device",
    "resolve_dtype",
    "resolve_family",
    "resolve_final_norm",
    "resolve_transformer_blocks",
    "supported_families",
]
