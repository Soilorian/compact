from dataclasses import dataclass

from compact.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class FeedbackDirectedMemoryAccess(MemoryAccess):
    demandMiss: bool
