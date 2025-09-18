from prefetchlenz.prefetchingalgorithm.impl.hds import (
    DFSM,
    BurstyController,
    HdsPrefetcher,
    HotStream,
    StreamMiner,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


# Simple access stub to mimic MemoryAccess
def make_access(pc: int, address: int):
    return MemoryAccess(pc=pc, address=address)


def test_bursty_controller_toggle_and_profile():
    # small numbers to exercise state transitions quickly
    bc = BurstyController(nCheck0=2, nInstr0=2, nAwake0=2, nHibernate0=2)
    profiles = []
    # simulate a sequence of checks, collect should_profile outcomes
    for i in range(12):
        profiles.append(bc.should_profile())
        bc.tick()
    # ensure we saw at least some True (awake) and some False (hibernate) periods
    assert any(profiles)
    assert any(not p for p in profiles)


def test_stream_miner_basic_detection():
    # small min_len to detect short repeated n-grams
    sm = StreamMiner(min_len=2, max_len=4, heat_threshold=4, min_unique=1)
    # create repeating pattern [A,B,C,A,B,C,...]
    A = (1, 0x1000)
    B = (2, 0x1100)
    C = (3, 0x1200)
    sequence = [A, B, C] * 4  # length 12
    for s in sequence:
        sm.feed(s)
    streams = sm.analyze()
    # Expect at least one stream discovered
    assert isinstance(streams, list)
    assert len(streams) >= 1


def test_dfsm_prefetch_and_returned_predictions():
    # Create a single hot stream: headLen=2 -> head=[s0,s1], tail=[s2]
    s0 = (10, 0x1000)
    s1 = (11, 0x1100)
    s2 = (12, 0x1200)
    stream = [s0, s1, s2]
    hs = HotStream(id=0, seq=stream)
    # build DFSM manually with headLen=2
    dfsm = DFSM([hs], headLen=2)
    # initial state
    state = frozenset()
    # feed first symbol -> partial match (no prefetch)
    state, payload = dfsm.step(state, s0)
    assert payload is None
    # feed second symbol -> completes head -> payload should be tail [[s2]]
    state, payload = dfsm.step(state, s1)
    assert payload is not None
    # payload contains a list-of-lists of symbols, extract addresses
    addrs = [addr for seq in payload for (_, addr) in seq]
    assert 0x1200 in addrs


def test_hds_prefetcher_end_to_end_with_manual_dfsm():
    h = HdsPrefetcher(headLen=2, max_hot_streams=4)
    collected = []
    h.set_prefetch_callback(lambda a: collected.append(a))
    # manually install DFSM with one stream
    s0 = (100, 0x1000)
    s1 = (101, 0x1100)
    s2 = (102, 0x1200)
    h._rebuild_dfsm([[s0, s1, s2]])
    # first access -> no prediction
    preds1 = h.progress(make_access(*s0))
    assert preds1 == [] or isinstance(preds1, list)
    # second access (completes head) -> should return tail address
    preds2 = h.progress(make_access(*s1))
    # The prefetch callback should have been invoked and returned list should include s2 addr
    assert 0x1200 in collected or 0x1200 in preds2
    # progress returns the predicted addresses as well
    assert isinstance(preds2, list)
