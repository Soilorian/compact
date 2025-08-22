import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm")

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess


# -----------------------------
# TCP data structures
# -----------------------------
@dataclass
class CorrelationEntry:
    """
    Successor frequency table for a given 'tag' (we use line address as tag).
    Keeps up to `max_succ` most frequent successors with integer counts.
    """

    max_succ: int
    freq: Counter = field(default_factory=Counter)
    total: int = 0

    def update(self, nxt: int, w: int = 1) -> None:
        """Increment correlation count tag->nxt with weight w; evict low-count tails."""
        self.freq[nxt] += w
        self.total += w
        # Bound table size by evicting least-common successors
        if len(self.freq) > self.max_succ:
            # drop the successor with the smallest count (ties arbitrary)
            victim, _ = min(self.freq.items(), key=lambda kv: kv[1])
            del self.freq[victim]

    def predict(self, tolerance: float, top_k: int) -> List[int]:
        """
        Return up to top_k successors whose probability is within `tolerance`
        of the best. Tolerance is absolute probability gap (0..1).
        """
        if not self.freq:
            return []
        # Convert to probabilities
        items = [(s, c / float(self.total)) for s, c in self.freq.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        best = items[0][1]
        keep: List[int] = []
        for s, p in items:
            if p >= best - tolerance:
                keep.append(s)
            else:
                break
            if len(keep) >= top_k:
                break
        return keep


@dataclass
class CorrelationTable:
    """
    Main tag->successors map with a hard capacity. On overflow, evict the entry
    with the smallest total count (cheap heuristic).
    """

    capacity: int
    max_succ_per_tag: int
    table: Dict[int, CorrelationEntry] = field(default_factory=dict)

    def _admit(self, tag: int) -> CorrelationEntry:
        if tag in self.table:
            return self.table[tag]
        if len(self.table) >= self.capacity:
            # Evict weakest entry by total counts
            victim = min(self.table.items(), key=lambda kv: kv[1].total)[0]
            del self.table[victim]
        ce = CorrelationEntry(max_succ=self.max_succ_per_tag)
        self.table[tag] = ce
        return ce

    def update(self, tag: int, successor: int, weight: int = 1) -> None:
        ce = self._admit(tag)
        ce.update(successor, w=weight)

    def predict(self, tag: int, tolerance: float, top_k: int) -> List[int]:
        ce = self.table.get(tag)
        if not ce:
            return []
        return ce.predict(tolerance=tolerance, top_k=top_k)


# -----------------------------
# TCP Prefetcher (PrefetchAlgorithm)
# -----------------------------
@dataclass
class TCPPrefetchAlgorithm(PrefetchAlgorithm):
    """
    Tag Correlating Prefetcher (TCP).

    This implementation models each last-miss tag (we use line address as the tag)
    as a key into a correlation table of likely next tags. On each miss, it
    strengthens the correlation from the previous tag to the current tag and,
    on prediction, returns the most likely successors of the current tag. It
    supports (a) multi-candidate prediction via probability tolerance, (b) a
    configurable prefetch degree (top_k), (c) chaining to follow the correlation
    graph several steps ahead, and (d) extra positive reinforcement for
    prefetch-hit events.  Inspired by the Tag Correlating Prefetchers design.  :contentReference[oaicite:1]{index=1}
    """

    # Geometry / interpretation
    line_size_bits: int = 6  # 64B lines by default (addr >> 6 -> line address)
    use_per_pc_context: bool = False  # if True, keep separate "last tag" per PC

    # Correlation-table sizing
    table_capacity: int = 1 << 15  # total number of source tags tracked
    max_succ_per_tag: int = 8  # bound successors per source tag

    # Prediction behavior
    top_k: int = 4  # max number of addresses returned
    tolerance: float = 0.05  # keep successors within abs prob gap of best
    chain_depth: int = 2  # 1 = no chaining, 2 = one hop, etc.
    avoid_duplicates: bool = True  # do not return repeated addresses in a batch

    # Training behavior
    prefetch_hit_boost: int = 2  # extra weight if the access was a prefetch hit
    base_weight: int = 1  # base update weight
    decay_every: int = 0  # 0 disables; >0: apply light decay every N updates
    decay_factor: float = 0.99  # multiplicative decay to slowly forget

    # Internal state
    corr: CorrelationTable = field(init=False)
    _prev_tag_global: Optional[int] = field(default=None, init=False)
    _prev_tag_per_pc: Dict[int, Optional[int]] = field(
        default_factory=lambda: defaultdict(lambda: None), init=False
    )
    _updates: int = field(default=0, init=False)

    def init(self):
        """Initialize correlation table and book-keeping."""
        self.corr = CorrelationTable(
            capacity=self.table_capacity, max_succ_per_tag=self.max_succ_per_tag
        )
        self._prev_tag_global = None
        self._prev_tag_per_pc.clear()
        self._updates = 0
        logger.info(
            "TCP init: cap=%d succ/tag=%d top_k=%d tol=%.3f chain=%d",
            self.table_capacity,
            self.max_succ_per_tag,
            self.top_k,
            self.tolerance,
            self.chain_depth,
        )

    # --- helpers ---
    def _extract(self, access: "MemoryAccess") -> Tuple[int, int]:
        """Extract (pc, address) robustly from MemoryAccess."""
        pc = getattr(access, "pc", None)
        if pc is None:
            pc = getattr(access, "PC", None)
        addr = getattr(access, "address", None)
        if addr is None:
            addr = getattr(access, "addr", None)
        if pc is None or addr is None:
            raise ValueError("MemoryAccess missing pc/address fields")
        return int(pc), int(addr)

    def _tag(self, addr: int) -> int:
        """Return the 'tag' we correlate on. We use the line address (addr >> line_size_bits)."""
        return addr >> self.line_size_bits

    def _addr_from_tag(self, tag: int) -> int:
        """Reconstruct a canonical byte address from a tag (lower bits zeroed)."""
        return tag << self.line_size_bits

    def _get_prev_tag(self, pc: int) -> Optional[int]:
        return (
            self._prev_tag_per_pc[pc]
            if self.use_per_pc_context
            else self._prev_tag_global
        )

    def _set_prev_tag(self, pc: int, tag: int) -> None:
        if self.use_per_pc_context:
            self._prev_tag_per_pc[pc] = tag
        else:
            self._prev_tag_global = tag

    def _maybe_decay(self):
        """Apply light decay to the whole table to gradually forget old correlations."""
        if self.decay_every <= 0:
            return
        self._updates += 1
        if self._updates % self.decay_every != 0:
            return
        # Apply decay: multiply all counts by decay_factor (>=0.0, <=1.0)
        # We keep integers by re-rounding; tiny entries may vanish over time.
        for ce in self.corr.table.values():
            new_freq = Counter()
            new_total = 0
            for s, c in ce.freq.items():
                dc = int(max(0, round(c * self.decay_factor)))
                if dc > 0:
                    new_freq[s] = dc
                    new_total += dc
            ce.freq = new_freq
            ce.total = new_total

    # --- PrefetchAlgorithm core ---
    def progress(self, access: "MemoryAccess", prefetch_hit: bool) -> List[int]:
        """
        Update correlations using the previous tag -> current tag transition
        and return predicted future addresses starting from the *current* tag.
        """
        pc, addr = self._extract(access)
        cur_tag = self._tag(addr)

        # 1) Training: strengthen correlation (prev_tag -> cur_tag).
        prev_tag = self._get_prev_tag(pc)
        if prev_tag is not None:
            w = self.base_weight + (self.prefetch_hit_boost if prefetch_hit else 0)
            self.corr.update(prev_tag, cur_tag, weight=w)
            self._maybe_decay()

        # 2) Update "previous" context with current tag.
        self._set_prev_tag(pc, cur_tag)

        # 3) Prediction from current tag (chain along correlation graph).
        predictions: List[int] = []
        seen_tags = set()

        frontier = [cur_tag]
        steps = max(1, self.chain_depth)
        budget = self.top_k

        for _ in range(steps):
            if not frontier or budget <= 0:
                break
            next_frontier: List[int] = []
            for src_tag in frontier:
                # Score successors within tolerance
                succs = self.corr.predict(
                    src_tag, tolerance=self.tolerance, top_k=budget
                )
                for s in succs:
                    if self.avoid_duplicates:
                        if s in seen_tags or s == cur_tag:
                            continue
                    seen_tags.add(s)
                    predictions.append(self._addr_from_tag(s))
                    budget -= 1
                    next_frontier.append(s)
                    if budget <= 0:
                        break
                if budget <= 0:
                    break
            frontier = next_frontier

        return predictions[: self.top_k]

    def close(self):
        """No persistent resources to release; keep for interface symmetry."""
        logger.info("TCP closed")
        return
