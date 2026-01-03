from dataclasses import dataclass
from typing import Optional

from compact.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class StoreOrderedMemoryAccess(MemoryAccess):
    isWrite: bool = False
    tid: Optional[int] = None
