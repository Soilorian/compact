import logging
from abc import ABC, abstractmethod

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm")


class PrefetchAlgorithm(ABC):
    """Interface for prefetching algorithms."""

    @abstractmethod
    def init(self):
        """Initialize any state before simulation begins."""
        pass

    @abstractmethod
    def progress(self, access: MemoryAccess, prefetch_hit: bool):
        """
        Process a single memory access.

        Args:
            access (MemoryAccess): The current memory access.
            prefetch_hit (bool): Whether the memory access is prefetched.

        Returns:
            List[address]: Predicted future addresses to prefetch.
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up any state after simulation ends."""
        pass
