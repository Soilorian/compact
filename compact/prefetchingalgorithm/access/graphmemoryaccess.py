from dataclasses import dataclass

from compact.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class GraphMemoryAccess(MemoryAccess):
    accessLatency: int
