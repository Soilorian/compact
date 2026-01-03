from dataclasses import dataclass

from compact.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class LinearMemoryAccess(MemoryAccess):
    loaded_pointer: bool
