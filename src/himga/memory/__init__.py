"""Memory module: abstract interface and built-in implementations."""

from himga.memory.base import BaseMemory
from himga.memory.null import NullMemory

__all__ = ["BaseMemory", "NullMemory"]
