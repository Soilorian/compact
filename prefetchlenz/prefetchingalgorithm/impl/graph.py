# file: prefetchers/graph_prefetcher.py
from __future__ import annotations

import heapq
import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

# If your interface lives elsewhere, adjust the import:
# from .prefetch_interface import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.graph")


@dataclass
class GraphPrefetcher(PrefetchAlgorithm):
    """
    Graph Prefetcher (software/simulator implementation).

    High-level idea (inspired by 'Graph Prefetching Using Data Structure Knowledge'):
      - Build a directed graph online where nodes are accessed addresses and edges
        represent observed transitions A -> B in the access stream.
      - Keep edge frequencies and prefer the most probable successors.
      - On an *event* (default: cache miss), issue prefetches for the top successors
        of the current address, and optionally their top successors (2-hop lookahead).
      - Cap total requests by a configurable prefetch degree; avoid duplicates.

    Why this fits your simulator:
      - Requires only the address stream you already have in `progress(...)`.
      - No need to dereference app pointers or peek into memory values.
      - Works for linked lists, trees, graphs, hash-chains, B-trees, etc., once
        short prefixes of transitions have been observed.

    Notes:
      - If you *do* have pointer values in `MemoryAccess` (e.g., `loaded_value` for
        a load), you can feed that to a custom hook to strengthen edges immediately.
      - This implementation is fully online and non-blocking for the simulator.

    Reference: Graph-based prefetching approach, learned over address transitions. :contentReference[oaicite:1]{index=1}
    """

    # ========================= Tunables =========================
    trigger_on_miss_only: bool = True
    max_successors_per_node: int = 8  # bound memory for per-node fan-out
    min_edge_support: int = 2  # minimum times we've seen A->B to use it
    lookahead_hops: int = 2  # 1 or 2 is typical for balance
    prefetch_degree: int = 8  # max prefetches to issue per trigger
    aging_half_life_events: Optional[int] = 50000
    """
    If set, edges are gradually aged so stale paths fade out. Half-life is counted
    in number of accesses (events). None disables aging.
    """

    # ========================= Internal state =========================
    # transitions[A] = Counter({B: count, ...})
    transitions: Dict[int, Counter] = field(
        default_factory=lambda: defaultdict(Counter), init=False
    )
    # Keep a compact top-k view per node for quick selection:
    # topk[A] = list[(count, B)] as a max-heap-like sorted list (we maintain sorted tuples)
    topk: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict, init=False)

    # last address in global stream (graph is built on global order by default)
    last_addr: Optional[int] = field(default=None, init=False)

    # Outstanding prefetches to avoid duplicates until they complete/are touched
    outstanding: Set[int] = field(default_factory=set, init=False)

    # Global event counter (for optional aging)
    events: int = field(default=0, init=False)

    # Optional: cap total distinct nodes/edges if desired (None = unlimited)
    max_nodes: Optional[int] = None
    max_edges: Optional[int] = None
    node_count: int = field(default=0, init=False)
    edge_count: int = field(default=0, init=False)

    # For light dedup of topK recomputations
    _dirty_nodes: Set[int] = field(default_factory=set, init=False)

    def init(self):
        self.transitions = defaultdict(Counter)
        self.topk = {}
        self.last_addr = None
        self.outstanding = set()
        self.events = 0
        self.node_count = 0
        self.edge_count = 0
        self._dirty_nodes = set()
        logger.debug("GraphPrefetcher initialized.")

    def close(self):
        logger.debug(
            "GraphPrefetcher closing. Nodes=%d, Edges~=%d",
            len(self.transitions),
            sum(len(c) for c in self.transitions.values()),
        )

    # ========================= Public API =========================
    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Update the transition graph and (on event) return prefetch addresses.
        """
        addr = getattr(access, "address", None)
        if addr is None:
            return []

        self.events += 1

        # 1) Update the transition graph with the last->current edge
        if self.last_addr is not None and self.last_addr != addr:
            self._observe_edge(self.last_addr, addr)

        # 2) Decide whether this access triggers prefetching
        triggered = self._should_trigger(access, prefetch_hit)
        self.last_addr = addr

        if not triggered:
            return []

        # 3) Generate prefetch candidates by walking graph successors
        candidates = self._select_successors(
            addr, self.prefetch_degree, self.lookahead_hops
        )

        # 4) Filter duplicates and mark outstanding
        issued: List[int] = []
        for c in candidates:
            if c not in self.outstanding and c != addr:
                issued.append(c)
                self.outstanding.add(c)
                if len(issued) >= self.prefetch_degree:
                    break

        if issued:
            logger.debug("GraphPrefetcher: addr=%d -> prefetch=%s", addr, issued)
        return issued

    # Optionally call when a prefetched line is confirmed/installed
    def prefetch_completed(self, addr: int):
        self.outstanding.discard(addr)

    # Optional hook if your cache tells us about hits to prefetched lines
    def notify_prefetch_hit(self, addr: int):
        self.outstanding.discard(addr)

    # ========================= Internals =========================
    def _should_trigger(self, access: MemoryAccess, prefetch_hit: bool) -> bool:
        if not self.trigger_on_miss_only:
            return True
        # Prefer simulator-provided "is_cache_miss" if available
        is_miss = getattr(access, "is_cache_miss", None)
        if is_miss is not None:
            return bool(is_miss) and not bool(prefetch_hit)
        # Fallback heuristic: trigger when not a prefetch-hit
        return not bool(prefetch_hit)

    def _observe_edge(self, a: int, b: int):
        """
        Observe transition a->b; update counts, aging, and maintain top-K per node.
        """
        # (Optional) limit the graph size
        if (
            self.max_nodes is not None
            and len(self.transitions) >= self.max_nodes
            and a not in self.transitions
        ):
            return

        counts = self.transitions[a]
        before = counts[b]
        counts[b] += 1

        if before == 0:
            # New edge
            self.edge_count += 1
            if self.max_edges is not None and self.edge_count > self.max_edges:
                # crude pruning: drop least-common edge in this node
                victim, vc = min(counts.items(), key=lambda kv: kv[1])
                counts[victim] -= 1
                if counts[victim] <= 0:
                    del counts[victim]
                    self.edge_count -= 1

        # Aging to keep graph fresh
        if self.aging_half_life_events:
            self._maybe_age_node(a)

        # Maintain top-K for quick selection later (lazy mark)
        self._dirty_nodes.add(a)

    def _maybe_age_node(self, a: int):
        """
        Exponential decay towards half-life in 'aging_half_life_events' accesses.
        Implemented as occasional downscale of all edges from node a.
        """
        # Cheap periodic aging: every half-life events, halve counts on dirty nodes
        if self.events % self.aging_half_life_events == 0:
            for node, counter in self.transitions.items():
                # halve (round up) to retain support ordering but fade old paths
                to_del = []
                for succ in list(counter.keys()):
                    newc = (counter[succ] + 1) // 2
                    if newc <= 0:
                        to_del.append(succ)
                    else:
                        counter[succ] = newc
                for s in to_del:
                    del counter[s]
            # Everything became stale; top-k views must be rebuilt lazily
            self.topk.clear()
            self._dirty_nodes.clear()

    def _refresh_topk_if_needed(self, a: int):
        if a not in self._dirty_nodes:
            return
        counts = self.transitions.get(a)
        if not counts:
            self.topk[a] = []
            self._dirty_nodes.discard(a)
            return
        # Keep only max_successors_per_node best successors
        best = heapq.nlargest(
            self.max_successors_per_node, counts.items(), key=lambda kv: kv[1]
        )
        # Store as a sorted list of (count, succ) descending by count
        self.topk[a] = [(c, s) for (s, c) in best if c >= self.min_edge_support]
        self._dirty_nodes.discard(a)

    def _select_successors(self, start: int, budget: int, hops: int) -> List[int]:
        """
        Best-first expansion from 'start' using per-node top-K lists.
        Returns a ranked list of candidate addresses (may exceed budget before dedup).
        """
        if budget <= 0:
            return []
        results: List[int] = []
        seen: Set[int] = set([start])

        # Priority queue of (-score, node, depth). Score = cumulative min-edge-count along path.
        # Using counts as a simple proxy for probability; more advanced scoring could combine log probs.
        pq: List[Tuple[int, int, int]] = []

        # Seed with immediate successors
        self._refresh_topk_if_needed(start)
        for c, succ in self.topk.get(start, []):
            if succ in seen:
                continue
            seen.add(succ)
            heapq.heappush(pq, (-c, succ, 1))
            results.append(succ)
            if len(results) >= budget:
                return results

        # Expand to next hops if allowed
        while pq and hops > 1 and len(results) < budget:
            negscore, node, depth = heapq.heappop(pq)
            if depth >= hops:
                continue
            self._refresh_topk_if_needed(node)
            for c, succ in self.topk.get(node, []):
                if succ in seen:
                    continue
                seen.add(succ)
                # Combine score conservatively (min along path) to avoid overestimating long chains
                newscore = min(-negscore, c)
                heapq.heappush(pq, (-newscore, succ, depth + 1))
                results.append(succ)
                if len(results) >= budget:
                    break

        return results
