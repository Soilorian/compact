"""
Dynamic Hot Data Stream Prefetching for General-Purpose Programs by Chilimbi et al.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from compact.prefetchingalgorithm.memoryaccess import MemoryAccess
from compact.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.hds")
logger.addHandler(logging.NullHandler())

Symbol = Tuple[int, int]  # (pc, addr)
StateElem = Tuple[int, int]  # (stream_id, seen) where 0 <= seen < headLen
State = FrozenSet[StateElem]


@dataclass(frozen=True)
class HotStream:
    """A hot data stream v = [(pc,addr), ...] with prefix length headLen."""

    id: int
    seq: List[Symbol]

    @property
    def head(self) -> List[Symbol]:
        # headLen will be injected by DFSM construction time as an attribute
        return self.seq[: self.headLen]  # type: ignore[attr-defined]

    @property
    def tail(self) -> List[Symbol]:
        return self.seq[self.headLen :]  # type: ignore[attr-defined]


# ----------------------- BurstyController -----------------------------------


class BurstyController:
    """
    Implements bursty tracing counters:
      - nCheck0, nInstr0 control a burst-period
      - nAwake0 periods of being 'awake' (instrument) then nHibernate0 'hibernate' periods.
    Use tick() per dynamic check and should_profile() to decide whether to record the
    reference in the profiling buffer for the current check.
    """

    def __init__(self, nCheck0: int, nInstr0: int, nAwake0: int, nHibernate0: int):
        assert nCheck0 >= 1 and nInstr0 >= 1
        self.nCheck0 = nCheck0
        self.nInstr0 = nInstr0
        self.nAwake0 = nAwake0
        self.nHibernate0 = nHibernate0

        self._phase_awake = True
        self._checks_left = nCheck0
        self._instr_left = 0
        self._period_left = nAwake0

    @property
    def is_awake(self) -> bool:
        return self._phase_awake

    def _flip_phase(self):
        self._phase_awake = not self._phase_awake
        self._period_left = self.nAwake0 if self._phase_awake else self.nHibernate0
        # Reset counters for a new period
        self._checks_left = (
            self.nCheck0 if self._phase_awake else (self.nCheck0 + self.nInstr0 - 1)
        )
        self._instr_left = self.nInstr0 if self._phase_awake else 1
        logger.debug("BurstyController: flipped phase -> awake=%s", self._phase_awake)

    def should_profile(self) -> bool:
        return self._phase_awake and self._instr_left > 0

    def tick(self):
        # One "check" passed
        if self._phase_awake:
            if self._instr_left > 0:
                self._instr_left -= 1
        else:
            # hibernating: occasional one-instruction instrumented
            if self._instr_left > 0:
                self._instr_left -= 1

        self._checks_left -= 1
        if self._checks_left == 0:
            self._period_left -= 1
            if self._period_left == 0:
                self._flip_phase()
            else:
                if self._phase_awake:
                    self._checks_left = self.nCheck0
                    self._instr_left = self.nInstr0
                else:
                    self._checks_left = self.nCheck0 + self.nInstr0 - 1
                    self._instr_left = 1


# ----------------------- StreamMiner ---------------------------------------


class StreamMiner:
    """
    Approximate hot-stream discovery via non-overlapping n-gram counts.
    Feed symbols via feed(sym). Call analyze() to return candidate streams
    as lists of Symbol. reset() clears internal buffer.
    """

    def __init__(
        self,
        min_len: int = 11,
        max_len: int = 64,
        heat_threshold: int = 8,
        min_unique: int = 2,
        max_profile_buffer: int = 50_000,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.heat_threshold = heat_threshold
        self.min_unique = min_unique
        self.max_profile_buffer = max_profile_buffer
        self.buffer: List[Symbol] = []

    def feed(self, sym: Symbol):
        if len(self.buffer) < self.max_profile_buffer:
            self.buffer.append(sym)

    def _nonoverlap_counts(self, seq: List[Symbol], L: int):
        counts: Dict[Tuple[Symbol, ...], int] = defaultdict(int)
        i = 0
        n = len(seq)
        while i + L <= n:
            s = tuple(seq[i : i + L])
            counts[s] += 1
            i += L
        return counts

    def analyze(self) -> List[List[Symbol]]:
        if not self.buffer:
            return []
        logger.info("HDS: StreamMiner analyzing %d samples", len(self.buffer))
        best: Dict[Tuple[Symbol, ...], int] = {}
        for start in (0, 1, 2):
            seq = self.buffer[start:]
            for L in range(self.min_len, min(self.max_len, len(seq)) + 1):
                counts = self._nonoverlap_counts(seq, L)
                for s, freq in counts.items():
                    if freq <= 1:
                        continue
                    if len(set(s)) < self.min_unique:
                        continue
                    heat = L * freq
                    if heat >= self.heat_threshold:
                        prev = best.get(s, 0)
                        if heat > prev:
                            best[s] = heat

        # De-subsumption: remove sequences that are contiguous substrings of larger chosen sequence
        streams = sorted(best.items(), key=lambda x: (-x[1], -len(x[0])))
        chosen: List[Tuple[Tuple[Symbol, ...], int]] = []
        for s, heat in streams:
            if any(self._is_subsequence(s, t[0]) for t in chosen):
                continue
            chosen.append((s, heat))

        result = [list(s) for (s, _) in chosen]
        logger.info("HDS: StreamMiner selected %d hot streams", len(result))
        return result

    @staticmethod
    def _is_subsequence(a: Tuple[Symbol, ...], b: Tuple[Symbol, ...]) -> bool:
        if len(a) > len(b):
            return False
        for i in range(len(b) - len(a) + 1):
            if b[i : i + len(a)] == list(a):
                return True
        return False

    def reset(self):
        self.buffer.clear()


# ----------------------- DFSM -----------------------------------------------


class DFSM:
    """
    Single DFSM matching prefixes of all hot streams in parallel.
    States are frozensets of (stream_id, seen).
    """

    def __init__(self, hot_streams: List[HotStream], headLen: int):
        self.headLen = headLen
        self.hot_streams = hot_streams
        for hs in self.hot_streams:
            object.__setattr__(hs, "headLen", headLen)  # attach for convenience

        self.alphabet: List[Symbol] = sorted(
            {
                hs.head[i]
                for hs in hot_streams
                for i in range(min(len(hs.head), headLen))
            }
        )
        self.start: State = frozenset()
        self.transitions: Dict[Tuple[State, Symbol], State] = {}
        self.prefetches: Dict[State, List[List[Symbol]]] = {}
        self._build()

    def _build(self):
        logger.info(
            "HDS: building DFSM for %d streams (headLen=%d)",
            len(self.hot_streams),
            self.headLen,
        )
        work: Deque[State] = deque([self.start])
        seen_states = {self.start}

        def advance(state: State, a: Symbol) -> State:
            next_elems: List[StateElem] = []
            for sid, n in state:
                hs = self.hot_streams[sid]
                if n < self.headLen and n < len(hs.head) and hs.head[n] == a:
                    next_elems.append((sid, n + 1))
            for hs in self.hot_streams:
                if len(hs.head) > 0 and hs.head[0] == a:
                    next_elems.append((hs.id, 1))
            return frozenset(next_elems)

        while work:
            s = work.popleft()
            for a in self.alphabet:
                ns = advance(s, a)
                if ns and (s, a) not in self.transitions:
                    self.transitions[(s, a)] = ns
                    if ns not in seen_states:
                        seen_states.add(ns)
                        work.append(ns)

        # annotate prefetch payloads for states that have completed heads
        for st in seen_states:
            payloads: List[List[Symbol]] = []
            for sid, n in st:
                if n >= self.headLen:
                    hs = self.hot_streams[sid]
                    if len(hs.tail) > 0:
                        payloads.append(hs.tail)
            if payloads:
                self.prefetches[st] = payloads

        logger.info(
            "HDS: DFSM built states=%d transitions=%d prefetch_states=%d",
            len(seen_states),
            len(self.transitions),
            len(self.prefetches),
        )

    def step(
        self, state: State, sym: Symbol
    ) -> Tuple[State, Optional[List[List[Symbol]]]]:
        ns = self.transitions.get((state, sym), frozenset())
        payload = self.prefetches.get(ns)
        return ns, payload


# ----------------------- HdsPrefetcher (orchestrator) -----------------------


class HdsPrefetcher(PrefetchAlgorithm):
    """
    Dynamic Hot Data Stream Prefetcher.
    Controls profiling, stream mining, DFSM building, and prefix matching.
    `progress(access, prefetch_hit)` returns the list of predicted addresses (ints)
    for that access. Also supports a prefetch callback via set_prefetch_callback(cb).
    """

    def __init__(
        self,
        headLen: int = 2,
        min_len: int = 11,
        max_len: int = 64,
        heat_threshold: int = 12,
        min_unique: int = 2,
        nCheck0: int = 11940,
        nInstr0: int = 60,
        nAwake0: int = 50,
        nHibernate0: int = 2450,
        max_hot_streams: int = 64,
    ):
        self.headLen = headLen
        self.controller = BurstyController(nCheck0, nInstr0, nAwake0, nHibernate0)
        self.miner = StreamMiner(min_len, max_len, heat_threshold, min_unique)
        self.dfsm: Optional[DFSM] = None
        self.state: State = frozenset()
        self.prefetch_cb = lambda addr: None
        self.max_hot_streams = max_hot_streams
        # Stats
        self._prefetches_issued = 0
        self._matches = 0

    def set_prefetch_callback(self, cb):
        """Set callback(cb: int) called per predicted address when prefetching."""
        self.prefetch_cb = cb

    def init(self):
        logger.info("HDS: init (headLen=%d)", self.headLen)
        self.miner.reset()
        self.dfsm = None
        self.state = frozenset()
        self._prefetches_issued = 0
        self._matches = 0

    def _symbolize(self, access: MemoryAccess) -> Symbol:
        return int(access.pc), int(access.address)

    def _rebuild_dfsm(self, streams: List[List[Symbol]]):
        ranked = sorted(streams, key=lambda s: (-len(s), s[0] if s else (0, 0)))[
            : self.max_hot_streams
        ]
        hot = [HotStream(id=i, seq=list(s)) for i, s in enumerate(ranked)]
        self.dfsm = DFSM(hot, self.headLen)
        self.state = frozenset()
        logger.info("HDS: rebuilt DFSM with %d streams", len(hot))

    def _maybe_profile(self, sym: Symbol):
        if self.controller.should_profile():
            self.miner.feed(sym)
            logger.debug("HDS: profiled symbol %s", str(sym))

    def _maybe_analyze_and_optimize(self):
        # Rebuild DFSM if none exists and miner has data
        if self.dfsm is None and len(self.miner.buffer) > 0:
            streams = self.miner.analyze()
            if streams:
                self._rebuild_dfsm(streams)
            self.miner.reset()

    def _prefetch_payloads(self, payloads: Iterable[List[Symbol]]) -> List[int]:
        """
        Issue prefetch callback for all payload symbols. Returns list of unique addresses issued.
        Deduplicated per invocation.
        """
        seen_block: Set[int] = set()
        issued_addrs: List[int] = []
        for seq in payloads:
            for _, addr in seq:
                if addr not in seen_block:
                    try:
                        self.prefetch_cb(addr)
                    except Exception:
                        logger.exception(
                            "HDS: prefetch_cb raised for addr=%s", hex(addr)
                        )
                    seen_block.add(addr)
                    issued_addrs.append(addr)
                    self._prefetches_issued += 1
                    logger.debug("HDS: prefetch issued addr=0x%X", addr)
        return issued_addrs

    def progress(self, access: MemoryAccess, prefetch_hit: bool = False) -> List[int]:
        """
        Should be called on every memory access.
        Returns a list of predicted addresses (may be empty).
        """
        sym = self._symbolize(access)
        predictions: List[int] = []

        # 1) profiling (sampled via bursty controller)
        self._maybe_profile(sym)

        # 2) matching/prefetching against DFSM if available
        if self.dfsm is not None:
            ns, payloads = self.dfsm.step(self.state, sym)
            if ns != self.state:
                logger.debug(
                    "HDS: DFSM state advanced from %s to %s on sym=%s",
                    str(self.state),
                    str(ns),
                    str(sym),
                )
                self.state = ns
            if payloads:
                self._matches += 1
                preds = self._prefetch_payloads(payloads)
                predictions.extend(preds)

        # 3) advance bursty counters and possibly rebuild DFSM when transitioning to hibernation
        self.controller.tick()

        # If we are (or just entered) hibernation and have buffered samples, try analyze+optimize
        if not self.controller.is_awake:
            self._maybe_analyze_and_optimize()

        # Optional reaction to prefetch_hit for adaptation (left as a hook)
        if prefetch_hit:
            logger.debug(
                "HDS: observed prefetch_hit for addr=%s",
                getattr(access, "address", None),
            )

        return predictions

    def close(self):
        logger.info(
            "HDS: close — matches=%d, prefetches_issued=%d",
            self._matches,
            self._prefetches_issued,
        )
        self.dfsm = None
        self.state = frozenset()
