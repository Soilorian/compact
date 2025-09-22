# tcp_prefetcher.py
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.tcp")
logger.addHandler(logging.NullHandler())


# -----------------------------
# TCP data structures
# -----------------------------
@dataclass
class CorrelationEntry:
    """
    Successor frequency table for a given 'tag' (line address).
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
            victim, _ = min(self.freq.items(), key=lambda kv: kv[1])
            del self.freq[victim]

    def predict(self, tolerance: float, top_k: int) -> List[int]:
        """
        Return up to top_k successors whose probability is within `tolerance`
        of the best. Tolerance is absolute probability gap (0..1).
        """
        if not self.freq or self.total <= 0:
            return []
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
    Main tag->successors mapping with hard capacity.
    On overflow, eject the entry with the smallest total count (cheap heuristic).
    """

    capacity: int
    max_succ_per_tag: int
    table: Dict[int, CorrelationEntry] = field(default_factory=dict)

    def _admit(self, tag: int) -> CorrelationEntry:
        if tag in self.table:
            return self.table[tag]
        if len(self.table) >= self.capacity:
            victim = min(self.table.items(), key=lambda kv: kv[1].total)[0]
            logger.debug(
                "CT: evicting tag=%d total=%d", victim, self.table[victim].total
            )
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
# TCP Prefetcher
# -----------------------------
@dataclass
class TCPPrefetchAlgorithm(PrefetchAlgorithm):
    """
    Tag Correlating Prefetcher (TCP).

    Operates on line tags (addr >> line_size_bits). On each access:
      - Train: strengthen prev_tag -> cur_tag (with optional boost on prefetch_hit).
      - Predict: return top correlated successors of cur_tag, optionally chaining.

    Key parameters:
      - line_size_bits: how many low bits to drop (e.g. 6 => 64B lines).
      - table_capacity, max_succ_per_tag: memory bounds for correlation table.
      - top_k, tolerance, chain_depth: prediction controls.
      - prefetch_hit_boost: extra weight for prefetched-and-used events.
      - decay_every, decay_factor: optional periodic decay of counters.
    """

    # Geometry / interpretation
    line_size_bits: int = 6  # 64B lines by default
    use_per_pc_context: bool = False

    # Correlation-table sizing
    table_capacity: int = 1 << 12
    max_succ_per_tag: int = 8

    # Prediction behavior
    top_k: int = 4
    tolerance: float = 0.05
    chain_depth: int = 2
    avoid_duplicates: bool = True

    # Training behavior
    prefetch_hit_boost: int = 2
    base_weight: int = 1
    decay_every: int = 0
    decay_factor: float = 0.99

    # Internal state (set in init)
    corr: CorrelationTable = field(init=False)
    _prev_tag_global: Optional[int] = field(default=None, init=False)
    _prev_tag_per_pc: Dict[int, Optional[int]] = field(
        default_factory=lambda: defaultdict(lambda: None), init=False
    )
    _updates: int = field(default=0, init=False)

    # -------------------------
    # Lifecycle
    # -------------------------
    def init(self):
        """Create a fresh correlation table and reset contexts."""
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

    def close(self):
        logger.info("TCP closed")

    # -------------------------
    # Helpers
    # -------------------------
    def _tag(self, addr: int) -> int:
        return addr >> self.line_size_bits

    def _addr_from_tag(self, tag: int) -> int:
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
        if self.decay_every <= 0:
            return
        self._updates += 1
        if self._updates % self.decay_every != 0:
            return
        logger.debug("TCP: applying decay")
        for ce in list(self.corr.table.values()):
            new_freq = Counter()
            new_total = 0
            for s, c in ce.freq.items():
                dc = int(max(0, round(c * self.decay_factor)))
                if dc > 0:
                    new_freq[s] = dc
                    new_total += dc
            ce.freq = new_freq
            ce.total = new_total

    # -------------------------
    # Core: process an access
    # -------------------------
    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        pc = int(access.pc)
        addr = int(access.address)
        cur_tag = self._tag(addr)

        # 1) training: prev_tag -> cur_tag
        prev_tag = self._get_prev_tag(pc)
        if prev_tag is not None:
            w = self.base_weight + (self.prefetch_hit_boost if prefetch_hit else 0)
            self.corr.update(prev_tag, cur_tag, weight=w)
            self._maybe_decay()

        # 2) update previous tag
        self._set_prev_tag(pc, cur_tag)

        # 3) prediction: chain from cur_tag
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
                succs = self.corr.predict(
                    src_tag, tolerance=self.tolerance, top_k=budget
                )
                for s in succs:
                    if self.avoid_duplicates and (s in seen_tags or s == cur_tag):
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
