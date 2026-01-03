from dataclasses import dataclass

from compact.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class EventTriggeredMemoryAccess(MemoryAccess):
    accessLatency: int
    bandwidthPressure: float
