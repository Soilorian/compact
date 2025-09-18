from dataclasses import dataclass

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class GraphMemoryAccess(MemoryAccess):
    accessLatency: int
