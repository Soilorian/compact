from collections import defaultdict, deque
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple


# MemoryAccess is expected to have .pc and .address fields (same as your loader)
@dataclass
class MemoryAccess:
    pc: int
    address: int


class BestOffsetPrefetcher:
    """
    Best-Offset prefetcher:
    - Maintain per-PC counters for candidate offsets.
    - Issue prefetches using the currently-best offset for that PC.
    - When a prefetch hit occurs, credit the offset that generated that prefetch.
    - Simple aging (periodic halving) and per-PC offset capacity (LFU eviction).
    """

    def __init__(
        self,
        max_offset: int = 16,  # consider offsets 1..max_offset
        track_offsets_per_pc: int = 32,  # how many offset counters to keep per pc
        degree: int = 1,  # number of prefetches: addr + offset, addr + 2*offset, ...
        aging_interval: int = 10000,  # accesses between counter halvings
        outstanding_limit: int = 4096,  # limit outstanding prefetch records
    ):
        self.max_offset = max(1, max_offset)
        self.track_offsets_per_pc = max(1, track_offsets_per_pc)
        self.degree = max(1, degree)
        self.aging_interval = max(1, aging_interval)
        self.outstanding_limit = max(1, outstanding_limit)

        # per-PC offset counters: pc -> {offset -> count}
        self.pc_offset_counts: Dict[int, Dict[int, int]] = defaultdict(dict)
        # best offset cache (for quick lookup): pc -> offset (computed lazily)
        self.pc_best_offset: Dict[int, int] = {}

        # outstanding prefetches mapping: address -> (pc_that_issued, offset_used)
        # when a prefetch_hit occurs for an address we check this map to credit the offset
        self.outstanding: Dict[int, Tuple[int, int]] = {}

        # simple FIFO queue to evict oldest outstanding when limit reached
        self.outstanding_queue: deque = deque()

        self.accesses_since_aging = 0

    def init(self):
        self.pc_offset_counts.clear()
        self.pc_best_offset.clear()
        self.outstanding.clear()
        self.outstanding_queue.clear()
        self.accesses_since_aging = 0

    def _ensure_offset_tracked(self, pc: int, offset: int):
        """Ensure offset entry exists for pc (create with count 0) and enforce per-pc capacity."""
        pc_map = self.pc_offset_counts[pc]
        if offset in pc_map:
            return
        # if capacity reached, evict LFU offset
        if len(pc_map) >= self.track_offsets_per_pc:
            # find least frequent offset and remove it
            victim = min(pc_map.items(), key=lambda kv: kv[1])[0]
            del pc_map[victim]
        pc_map[offset] = 0
        # best offset cached invalidation
        if pc in self.pc_best_offset:
            del self.pc_best_offset[pc]

    def _select_best_offset(self, pc: int) -> int:
        """Return best offset for pc (choose 1 if no data). Cache the result."""
        if pc in self.pc_best_offset:
            return self.pc_best_offset[pc]
        pc_map = self.pc_offset_counts.get(pc)
        if not pc_map:
            best = 1
        else:
            # pick offset with highest count; tie-breaker: smallest offset (prefers locality)
            best = min(sorted(pc_map.items(), key=lambda kv: (-kv[1], kv[0])))[0]
            # Explanation: sorting yields tuples (offset,count) -> use (-count, offset)
            # then min gives the offset with highest count, tie-broken by smaller offset
        self.pc_best_offset[pc] = best
        return best

    def _record_outstanding(self, addr: int, pc: int, offset: int):
        """Record a prefetch for later crediting; respect outstanding_limit."""
        if addr in self.outstanding:
            return
        self.outstanding[addr] = (pc, offset)
        self.outstanding_queue.append(addr)
        if len(self.outstanding_queue) > self.outstanding_limit:
            old = self.outstanding_queue.popleft()
            self.outstanding.pop(old, None)

    def _credit_offset_for_prefetch_hit(self, addr: int):
        """
        If addr was an outstanding prefetch, credit the corresponding pc/offset.
        Return True if credited.
        """
        info = self.outstanding.pop(addr, None)
        if info is None:
            return False
        pc, offset = info
        # remove from queue if still present (cheap: ignore if not present)
        try:
            self.outstanding_queue.remove(addr)
        except ValueError:
            pass
        # ensure offset tracked and increment count
        self._ensure_offset_tracked(pc, offset)
        self.pc_offset_counts[pc][offset] += 1
        # invalidate cached best offset for pc
        if pc in self.pc_best_offset:
            del self.pc_best_offset[pc]
        return True

    def _age_counters(self):
        """Halve all counters (integer division) to age out old patterns."""
        for pc, cmap in list(self.pc_offset_counts.items()):
            for off in list(cmap.keys()):
                cmap[off] = cmap[off] // 2
            # if a pc's map becomes empty, remove it to free memory
            if not cmap:
                del self.pc_offset_counts[pc]
                self.pc_best_offset.pop(pc, None)
            else:
                # maybe invalidate cached best offset
                self.pc_best_offset.pop(pc, None)

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process an access.
        - If prefetch_hit is True, credit the outstanding prefetch that brought this block (if any).
        - Select best offset for the PC, issue prefetches with configured degree.
        - Return list of addresses to prefetch.
        """
        self.accesses_since_aging += 1
        if self.accesses_since_aging >= self.aging_interval:
            self._age_counters()
            self.accesses_since_aging = 0

        addr = access.address
        pc = access.pc
        prefetches: List[int] = []

        # 1) credit offset if this access was a prefetch hit
        if prefetch_hit:
            # If this access was prefetched previously, outstanding map should contain it.
            credited = self._credit_offset_for_prefetch_hit(addr)
            # credited may be False if outstanding entry expired/evicted; we ignore that case.

        # 2) pick best offset for this PC (fallback to 1)
        best_offset = self._select_best_offset(pc)

        # clamp best_offset to allowed range
        if best_offset <= 0 or best_offset > self.max_offset:
            best_offset = 1

        # 3) issue degree prefetches: addr + best_offset * i for i=1..degree
        for i in range(1, self.degree + 1):
            tgt = addr + best_offset * i
            # record outstanding mapping to credit when/if a hit happens
            self._record_outstanding(tgt, pc, best_offset)
            prefetches.append(tgt)

        return prefetches

    def observe_other_offset_candidate(self, pc: int, offset: int, success: int = 1):
        """
        Optional API: allow external training/feedback for offsets (not required).
        Increment pc->offset by 'success' (useful when offline evaluating many offsets).
        """
        if offset <= 0 or offset > self.max_offset:
            return
        self._ensure_offset_tracked(pc, offset)
        self.pc_offset_counts[pc][offset] += success
        if pc in self.pc_best_offset:
            del self.pc_best_offset[pc]

    def close(self):
        self.pc_offset_counts.clear()
        self.pc_best_offset.clear()
        self.outstanding.clear()
        self.outstanding_queue.clear()
