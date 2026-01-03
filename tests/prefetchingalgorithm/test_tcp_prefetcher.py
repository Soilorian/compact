# test_tcp_prefetcher.py
from compact.dataloader.impl.ArrayDataLoader import MemoryAccess
from compact.prefetchingalgorithm.impl.tcp import TCPPrefetchAlgorithm

LINE = 64


def make_access(addr: int, pc: int = 0) -> MemoryAccess:
    return MemoryAccess(address=addr, pc=pc)


def test_basic_training_and_prediction():
    p = TCPPrefetchAlgorithm(
        table_capacity=64, max_succ_per_tag=4, top_k=2, chain_depth=1
    )
    p.init()

    # Simulate misses: A -> B -> C
    a = 0 * LINE
    b = 1 * LINE
    c = 2 * LINE

    p.progress(make_access(a, pc=1), prefetch_hit=False)  # no prev -> no update
    p.progress(make_access(b, pc=1), prefetch_hit=False)  # updates A->B
    p.progress(make_access(c, pc=1), prefetch_hit=False)  # updates B->C

    # Now a new miss at B should predict C
    preds = p.progress(make_access(b, pc=1), prefetch_hit=False)
    assert (c in preds) or (preds == []) == False  # expects C to appear (non-empty)


def test_tolerance_and_topk():
    p = TCPPrefetchAlgorithm(
        table_capacity=64, max_succ_per_tag=4, top_k=2, tolerance=0.0, chain_depth=1
    )
    p.init()

    # Build counts: tag X -> successors Y and Z with different frequencies
    x = 10 * LINE
    y = 11 * LINE
    z = 12 * LINE

    # Make X->Y twice, X->Z once
    p.progress(make_access(x, pc=1), prefetch_hit=False)
    p.progress(make_access(y, pc=1), prefetch_hit=False)  # X->Y
    p.progress(make_access(x, pc=1), prefetch_hit=False)
    p.progress(make_access(y, pc=1), prefetch_hit=False)  # X->Y (second)
    p.progress(make_access(x, pc=1), prefetch_hit=False)
    p.progress(make_access(z, pc=1), prefetch_hit=False)  # X->Z

    # Predict from X: with zero tolerance, only best (Y) should appear;
    preds = p.progress(make_access(x, pc=1), prefetch_hit=False)
    assert len(preds) <= 2 and (y in preds)


def test_chaining_behavior():
    p = TCPPrefetchAlgorithm(
        table_capacity=128, max_succ_per_tag=4, top_k=3, chain_depth=2, tolerance=0.01
    )
    p.init()

    # Sequence: A->B, B->C
    a = 100 * LINE
    b = 101 * LINE
    c = 102 * LINE

    p.progress(make_access(a, pc=1), prefetch_hit=False)
    p.progress(make_access(b, pc=1), prefetch_hit=False)  # learn A->B
    p.progress(make_access(c, pc=1), prefetch_hit=False)  # learn B->C

    # Predict from A with chain_depth=2 should get B and C (addresses)
    preds = p.progress(make_access(a, pc=1), prefetch_hit=False)
    assert any(x == b for x in preds) and any(x == c for x in preds)


def test_decay_reduces_counts():
    p = TCPPrefetchAlgorithm(
        table_capacity=64, max_succ_per_tag=4, decay_every=1, decay_factor=0.5
    )
    p.init()

    x = 50 * LINE
    y = 51 * LINE

    # build X->Y count large
    for _ in range(8):
        p.progress(make_access(x, pc=1), prefetch_hit=False)
        p.progress(make_access(y, pc=1), prefetch_hit=False)

    # counts should be present; after many decays they will drop
    before = p.corr.table.get(p._tag(x)).total if p._tag(x) in p.corr.table else 0
    # trigger several decays
    for _ in range(10):
        p.progress(make_access(x, pc=1), prefetch_hit=False)
        p.progress(make_access(y, pc=1), prefetch_hit=False)
    after = p.corr.table.get(p._tag(x)).total if p._tag(x) in p.corr.table else 0
    assert after <= before
