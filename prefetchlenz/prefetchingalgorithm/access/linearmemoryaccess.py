from dataclasses import dataclass

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class LinearMemoryAccess(MemoryAccess):
    loaded_pointer: bool
