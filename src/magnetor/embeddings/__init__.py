"""Embedding layer (Spec Section 7.1).

Turns text into vectors through a provider-agnostic :class:`Embedder` boundary.
The asymmetric ``embed_query`` / ``embed_documents`` split lets a provider tag
questions and passages differently, which the spec calls out as the likely fix
for the original unreachable similarity threshold.
"""

from magnetor.embeddings.base import Embedder
from magnetor.embeddings.voyage import VoyageEmbedder

__all__ = ["Embedder", "VoyageEmbedder"]
