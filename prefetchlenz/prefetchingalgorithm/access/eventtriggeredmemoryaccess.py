from dataclasses import dataclass

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class EventTriggeredMemoryAccess(MemoryAccess):
    accessLatency: int
    bandwidthPressure: float
