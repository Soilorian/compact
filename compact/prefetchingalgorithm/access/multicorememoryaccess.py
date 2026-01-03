from dataclasses import dataclass

from compact.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class MulticoreMemoryAccess(MemoryAccess):
    cpu: int
