"""gently-deabe — PyTorch port of the 3D-RCAN DeAbe / Decon / Expan pipeline.

Public API:
    RCAN                 — the architecture
    convert              — Keras .hdf5 -> PyTorch state_dict
    apply, normalize     — tiled 3D inference + percentile-stretch normalize
    Pipeline             — multi-step orchestration (DeAbe -> Decon -> Expan)
    ModelRegistry        — named-model -> weights-path map (loaded lazily)
"""
from .model import RCAN, RCAB, ChannelAttention, ResidualGroup
from .convert import convert
from .inference import apply, normalize
from .pipeline import Pipeline, PipelineStep
from .registry import ModelRegistry, ModelEntry

__all__ = [
    "RCAN",
    "RCAB",
    "ChannelAttention",
    "ResidualGroup",
    "convert",
    "apply",
    "normalize",
    "Pipeline",
    "PipelineStep",
    "ModelRegistry",
    "ModelEntry",
]

__version__ = "0.1.0"
