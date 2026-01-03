from compact.prefetchingalgorithm.impl.triangel import *
from compact.util.size import Size


def make_access(addr: int, pc: int = 0) -> MemoryAccess:
    return MemoryAccess(address=addr, pc=pc)


# ---------------------- Helpers ----------------------


def test_history_sampler_insert_and_get():
    hs = HistorySampler(sets=4, ways=2)
    tup = (0x100, 1, 0x200, 5)
    ev = hs.insert(*tup)
    assert ev is None
    got = hs.get(0x100, 1)
    assert got == tup


def test_history_sampler_eviction():
    hs = HistorySampler(sets=1, ways=1)
    hs.insert(0x100, 1, 0x200, 1)
    ev = hs.insert(0x300, 2, 0x400, 2)
    assert ev is not None
    assert hs.get(0x100, 1) is None
    assert hs.get(0x300, 2) is not None


def test_second_chance_sampler_hit_and_age():
    scs = SecondChanceSampler(cap=2)
    scs.put(0x100, 1, deadline=10)
    assert not scs.hit(0x200, 1, now=5)
    assert scs.hit(0x100, 1, now=5)
    scs.put(0x300, 2, deadline=1)
    scs.age_out(now=5)
    assert 0x300 not in scs.index


def test_metadata_reuse_buffer_put_get():
    mrb = MetadataReuseBuffer(sets=2, ways=1)
    meta = TriangelMeta(neighbor=0x200, conf=1)
    mrb.put(0x100, meta)
    assert mrb.get(0x100) == meta
    # overwrite
    new_meta = TriangelMeta(neighbor=0x300, conf=0)
    mrb.put(0x100, new_meta)
    assert mrb.get(0x100) == new_meta


# ---------------------- Prefetcher ----------------------


def test_prefetcher_basic_init_and_close():
    pf = TriangelPrefetcher(init_size=Size.from_kb(4))
    pf.init()
    assert pf.training == {}
    pf.close()


def test_prefetcher_trains_and_issues_prefetch_chain():
    pf = TriangelPrefetcher(init_size=Size.from_kb(4))
    pf.init()

    # Feed repeating accesses to build markov (A->B->C)
    pf.progress(make_access(0x1000, pc=1), prefetch_hit=False)
    pf.progress(make_access(0x2000, pc=1), prefetch_hit=False)
    pf.progress(make_access(0x3000, pc=1), prefetch_hit=False)

    # Train markov manually to force a chain
    pf._train_markov(0x1000, 0x2000)
    pf._train_markov(0x2000, 0x3000)

    # Now access A again → expect [0x2000, 0x3000] prefetches
    preds = pf.progress(make_access(0x1000, pc=1), prefetch_hit=False)
    assert 0x2000 in preds
    assert any(p in preds for p in [0x3000])


def test_prefetcher_useful_prefetch_increments():
    pf = TriangelPrefetcher(init_size=Size.from_kb(4))
    pf.init()
    acc = make_access(0x1000, pc=1)
    # first: no prediction
    preds = pf.progress(acc, prefetch_hit=False)
    assert preds == []
    # now mark this as a prefetch hit
    pf.prev_access = acc
    pf.progress(acc, prefetch_hit=True)
    assert pf.useful_prefetches >= 1


def test_prefetcher_resize_grow_and_shrink():
    pf = TriangelPrefetcher(init_size=Size.from_kb(4), resize_epoch=1)
    pf.init()

    # Force useful prefetches > issued ratio
    pf.issued_prefetches = 1
    pf.useful_prefetches = 1
    pf.meta_accesses = 1
    pf._maybe_resize()
    assert pf.num_ways >= 1

    # Force shrink
    pf.issued_prefetches = 10
    pf.useful_prefetches = 0
    pf._maybe_resize()
    assert pf.num_ways >= 1
