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
    Metadata for a single address:
      neighbor    – correlated next address
      confidence  – 1-bit counter to avoid thrash
      score       – usefulness count for Hawkeye replacement
    """

    neighbor: int
    confidence: int = 1
    score: int = 0


class TriagePrefetcher(PrefetchAlgorithm):
    """
    Triage Prefetcher with on-chip metadata,
    Hawkeye-style replacement, and dynamic sizing.
    """

    def __init__(
        self,
        init_size: Size = Size.from_kb(512),
        min_size: Size = Size.from_kb(128),
        max_size: Size = Size.from_mb(2),
        resize_epoch: int = 50_000,
        grow_thresh: float = 0.1,
        shrink_thresh: float = 0.05,
    ):
        self.size = init_size
        self.min_size = min_size
        self.max_size = max_size

        # table: {address → metadata}
        self.table: dict[int, TriagePrefetcherMetaData] = {}

        # last access per PC (for training)
        self.last_access_per_pc: dict[int, int] = {}

        # stats for dynamic resizing
        self.meta_accesses = 0
        self.useful_prefetches = 0
        self.resize_epoch = resize_epoch
        self.grow_thresh = grow_thresh
        self.shrink_thresh = shrink_thresh

        logger.debug("TriagePrefetcher instantiated")

    def init(self, size: Optional[Size] = None):
        """
        Initialize or reset internal state.

        Args:
            size (Size, optional): override initial metadata capacity.
        """
        if size:
            self.size = size
        self.table.clear()
        self.last_access_per_pc.clear()
        self.meta_accesses = 0
        self.useful_prefetches = 0
        logger.info(f"Triage init: metadata capacity = {self.size}")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process one memory access, update metadata and maybe predict.

        Args:
            access (MemoryAccess): (address, PC)
            prefetch_hit (bool): whether this access was prefetched

        Returns:
            List[int]: list with at most one prefetch address
        """
        pc = access.pc
        addr = access.address
        preds: List[int] = []

        # 1) Train on PC-localized stream
        prev = self.last_access_per_pc.get(pc)

        if prev is not None and prefetch_hit:
            entry = self.table.get(prev)
            entry.score += 1
            self.useful_prefetches += 1
            logger.debug(
                f"prefetch hit! updating score {hex(addr)}→{hex(entry.neighbor)} (score={entry.score})"
            )

        if prev is not None:
            md = self.table.get(prev)
            # update or insert mapping prev → addr
            if md:
                if md.neighbor == addr:
                    md.confidence = 1
                else:
                    md.confidence -= 1
                    if md.confidence <= 0:
                        md.neighbor = addr
                        md.confidence = 1
            else:
                # if full, evict one
                if len(self.table) >= int(self.size):
                    self._evict_entry()
                self.table[prev] = TriagePrefetcherMetaData(neighbor=addr)
                logger.debug(f"New mapping: {hex(prev)}→{hex(addr)}")

        # 2) Prediction only if this PC is “trained” (we’ve seen it before)
        if prev is not None:
            entry = self.table.get(addr)
            if entry:
                preds.append(entry.neighbor)
                # score this entry only if the subsequent access missed (useful)
                if prefetch_hit:
                    entry.score += 1
                    self.useful_prefetches += 1
                logger.debug(
                    f"Predict {hex(addr)}→{hex(entry.neighbor)} (score={entry.score})"
                )

        # 3) Update stats and maybe resize
        self.meta_accesses += 1
        if self.meta_accesses >= self.resize_epoch:
            self._maybe_resize()
            self.meta_accesses = 0
            self.useful_prefetches = 0

        # 4) Update last access for this PC
        self.last_access_per_pc[pc] = addr
        return preds

    def _evict_entry(self):
        """
        Evict the entry with the lowest 'score' (Hawkeye-style).
        """
        # find the key with minimum score
        victim = min(self.table.items(), key=lambda kv: kv[1].score)[0]
        md = self.table.pop(victim)
        logger.info(f"Evicted {hex(victim)}→{hex(md.neighbor)} (score={md.score})")

    def _maybe_resize(self):
        """
        Grow/shrink metadata capacity based on usefulness ratio.
        """
        ratio = self.useful_prefetches / max(1, self.resize_epoch)
        old = int(self.size)
        if ratio > self.grow_thresh and int(self.size) < int(self.max_size):
            # grow by 1.5×, capped
            new_bytes = min(int(self.max_size), int(self.size) * 3 // 2)
            self.size = Size(new_bytes)
            logger.info(f"Growing table: {old}→{new_bytes} bytes (ratio={ratio:.3f})")
        elif ratio < self.shrink_thresh and int(self.size) > int(self.min_size):
            # shrink by 0.75×, floored
            new_bytes = max(int(self.min_size), int(self.size) * 3 // 4)
            self.size = Size(new_bytes)
            logger.info(f"Shrinking table: {old}→{new_bytes} bytes (ratio={ratio:.3f})")

    def close(self):
        """
        Final cleanup.
        """
        logger.info(f"Triage closed: final entries={len(self.table)}")
        self.table.clear()
        self.last_access_per_pc.clear()
