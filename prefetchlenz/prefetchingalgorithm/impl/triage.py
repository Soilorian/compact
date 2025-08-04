import logging
from dataclasses import dataclass
from typing import List, Optional

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm
from prefetchlenz.util.size import Size

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.triage")


@dataclass
class TriagePrefetcherMetaData:
    """
    Represents metadata for a single memory address in the Triage prefetcher.

    Attributes:
        neighbor (int): The next correlated address.
        confidence (int): A 1-bit confidence counter (0 or 1).
    """

    neighbor: int
    confidence: int = 1


class TriagePrefetcher(PrefetchAlgorithm):
    """
    Triage Prefetcher: A temporal prefetcher that identifies PC-localized address
    correlations and stores metadata entirely on-chip.

    This prefetcher uses a dynamic table indexed by the last accessed address per PC
    and predicts future accesses based on learned address correlations.
    """

    def __init__(self):
        """
        Initialize the Triage prefetcher state.
        """
        self.size: Optional[Size] = None
        self.table: dict[int, TriagePrefetcherMetaData] = {}
        self.last_access_per_pc: dict[int, int] = {}
        logger.debug("TriagePrefetcher instantiated")

    def init(self, size=Size.from_kb(512)):
        """
        Initialize internal table and metadata size.

        Args:
            size (Size): The maximum size (in bytes) of the metadata table.
        """
        self.size = size
        logger.info(f"TriagePrefetcher initialized with metadata size: {self.size}")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process a memory access and determine if a prefetch should be issued.

        Args:
            access (MemoryAccess): The memory access (address, PC).
            prefetch_hit (bool): Whether this access was already prefetched.

        Returns:
            List[int]: A list containing one predicted address to prefetch, or empty if none.
        """
        current_pc = access.pc
        current_addr = access.address
        predictions = []

        logger.debug(
            f"Progressing access: PC={hex(current_pc)}, Addr={hex(current_addr)}"
        )

        # Try to build a correlation (previous_addr → current_addr)
        previous_addr = self.last_access_per_pc.get(current_pc)

        if previous_addr is not None:
            entry = self.table.get(previous_addr)

            if entry:
                if entry.neighbor == current_addr:
                    entry.confidence = min(entry.confidence + 1, 1)
                    logger.debug(
                        f"Confirmed correlation ({hex(previous_addr)} → {hex(current_addr)})"
                    )
                else:
                    entry.confidence -= 1
                    logger.debug(f"Decaying confidence for {hex(previous_addr)}")
                    if entry.confidence <= 0:
                        logger.info(
                            f"Updating neighbor for {hex(previous_addr)} to {hex(current_addr)}"
                        )
                        entry.neighbor = current_addr
                        entry.confidence = 1
            else:
                if len(self.table) >= int(self.size):
                    self._evict_metadata()
                self.table[previous_addr] = TriagePrefetcherMetaData(
                    neighbor=current_addr
                )
                logger.info(
                    f"Inserted new metadata entry: {hex(previous_addr)} → {hex(current_addr)}"
                )

        # Predict from current address
        entry = self.table.get(current_addr)
        if entry:
            predictions.append(entry.neighbor)
            logger.debug(
                f"Prefetch prediction: {hex(current_addr)} → {hex(entry.neighbor)}"
            )

        # Update last address for this PC
        self.last_access_per_pc[current_pc] = current_addr
        return predictions

    def _evict_metadata(self):
        """
        Evict an entry from the metadata table using a simple FIFO policy.
        """
        evicted_key = next(iter(self.table))
        evicted_entry = self.table.pop(evicted_key)
        logger.warning(
            f"Evicted metadata entry: {hex(evicted_key)} → {hex(evicted_entry.neighbor)}"
        )

    def close(self):
        """
        Clean up internal state and log summary.
        """
        logger.info(
            f"TriagePrefetcher shutting down. Total metadata entries: {len(self.table)}"
        )
        self.table.clear()
        self.last_access_per_pc.clear()
