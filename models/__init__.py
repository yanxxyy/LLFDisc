"""Model definitions for LLFDisc."""

from .llfdisc import CAB, IEL, EnhancedLCA, LayerNorm, LLFDisc, SEBlock

# Backwards-compatible alias: earlier revisions of this code exported the
# network as ``EnhancedLYT``. The submodule attribute names are unchanged, so
# checkpoints saved under the old name load into ``LLFDisc`` without remapping.
EnhancedLYT = LLFDisc

__all__ = [
    "CAB",
    "IEL",
    "EnhancedLCA",
    "EnhancedLYT",
    "LayerNorm",
    "LLFDisc",
    "SEBlock",
]
