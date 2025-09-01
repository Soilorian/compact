from dataclasses import dataclass

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class FeedbackDirectedMemoryAccess(MemoryAccess):
    demandMiss: bool
