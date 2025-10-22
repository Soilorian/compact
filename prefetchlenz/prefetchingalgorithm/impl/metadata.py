"""
Efficient Metadata Management for Irregular Data Prefetching by Wu et al.
"""

import logging
from typing import Dict, List, Optional

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.lru import LruReplacementPolicy
from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.metadata")


class MetadataEntry:
    """
    Represents one metadata entry for a trigger PC.
    Tracks correlated target addresses with confidence counters.
    """

    def __init__(self, key: int):
        self.key = key
        self.correlations: Dict[int, int] = {}  # target_addr -> confidence
        self.use_count = 0

    def touch(self):
        """Increment use counter when prefetch hits are observed."""
        self.use_count += 1

    def update_correlation(self, target_addr: int):
        """
        Increment confidence for a target address.
        Confidence counters are capped at 255 (8-bit).
        """
        self.correlations[target_addr] = self.correlations.get(target_addr, 0) + 1
        if self.correlations[target_addr] > 255:
            self.correlations[target_addr] = 255
        logger.debug(
            "Updated correlation: trigger=%s target=0x%x conf=%d",
            self.key,
            target_addr,
            self.correlations[target_addr],
        )

    def top_targets(self, threshold: int = 2, max_preds: int = 4) -> List[int]:
        """
        Select top-N correlated target addresses above confidence threshold.
        """
        sorted_corrs = sorted(
            self.correlations.items(), key=lambda kv: kv[1], reverse=True
        )
        return [addr for addr, conf in sorted_corrs if conf >= threshold][:max_preds]


class MetadataPrefetcher(PrefetchAlgorithm):
    """
    Efficient Metadata Management Prefetcher.

    - Uses trigger PCs as metadata keys.
    - Stores correlations between trigger and next addresses.
    - Confidence counters decide which prefetches to issue.
    - Metadata entries are stored in a pluggable set-associative Cache.
    """

    def __init__(
        self,
        num_sets: int = 64,
        num_ways: int = 8,
        replacement_policy_cls=LruReplacementPolicy,
    ):
        self.cache = Cache(
            num_sets=num_sets,
            num_ways=num_ways,
            replacement_policy_cls=replacement_policy_cls,
        )
        self.last_addr_per_pc: Dict[int, int] = {}

    def init(self):
        self.cache.flush()
        self.last_addr_per_pc.clear()
        logger.info("MetadataPrefetcher initialized.")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        pc = int(access.pc)
        addr = int(access.address)

        # 1. Lookup or allocate metadata entry
        entry: Optional[MetadataEntry] = self.cache.get(pc)
        if entry is None:
            entry = MetadataEntry(pc)
            self.cache.put(pc, entry)

        # 2. Update correlations (prev -> current)
        if pc in self.last_addr_per_pc:
            prev_addr = self.last_addr_per_pc[pc]
            if prev_addr != addr:
                entry.update_correlation(addr)

        self.last_addr_per_pc[pc] = addr

        # 3. Predict future addresses
        preds = entry.top_targets(threshold=2, max_preds=4)
        preds = [p for p in preds if p != addr]
        if preds:
            logger.debug(
                "Prefetch triggered: pc=%s addr=0x%x preds=%s", pc, addr, preds
            )

        # 4. Feedback: boost usefulness
        if prefetch_hit:
            entry.touch()

        return preds

    def close(self):
        self.cache.flush()
        self.last_addr_per_pc.clear()
        logger.info("MetadataPrefetcher closed.")
