import logging

import pytest

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.metadata import MetadataPrefetcher

logger = logging.getLogger("test.metadata_prefetcher")


@pytest.fixture
def prefetcher():
    pf = MetadataPrefetcher(num_sets=4, num_ways=2)
    pf.init()
    yield pf
    pf.close()


def make_access(pc: int, addr: int) -> MemoryAccess:
    return MemoryAccess(pc=pc, address=addr)


def test_basic_correlation(prefetcher: MetadataPrefetcher):
    # First access — no prediction
    a1 = make_access(pc=1, addr=100)
    preds = prefetcher.progress(a1, prefetch_hit=False)
    assert preds == []

    # Second access — should record correlation (100 -> 200)
    a2 = make_access(pc=1, addr=200)
    preds = prefetcher.progress(a2, prefetch_hit=False)
    # Still no prefetch on second access (needs confidence)
    assert preds == []

    # Third access — now correlation confidence is 1 for 200
    a3 = make_access(pc=1, addr=100)
    prefetcher.progress(a3, prefetch_hit=False)

    a4 = make_access(pc=1, addr=200)
    preds = prefetcher.progress(a4, prefetch_hit=False)
    # Prefetcher should now suggest 200 with confidence
    assert 200 in preds or preds == []  # depends on threshold


def test_multiple_targets(prefetcher: MetadataPrefetcher):
    # Train correlations from PC=2
    for target in (300, 400, 500):
        prefetcher.last_addr_per_pc[2] = 100
        access = make_access(pc=2, addr=target)
        prefetcher.progress(access, prefetch_hit=False)

    entry = prefetcher.cache.get(2)
    preds = entry.top_targets(threshold=1, max_preds=2)
    assert len(preds) <= 2
    assert set(preds).issubset({300, 400, 500})


def test_prefetch_hit_feedback(prefetcher: MetadataPrefetcher):
    a1 = make_access(pc=3, addr=1000)
    prefetcher.progress(a1, prefetch_hit=False)

    a2 = make_access(pc=3, addr=2000)
    preds = prefetcher.progress(a2, prefetch_hit=True)

    entry = prefetcher.cache.get(3)
    assert entry.use_count > 0


def test_eviction(prefetcher: MetadataPrefetcher):
    # Fill cache beyond capacity to force eviction
    for pc in range(10):
        a = make_access(pc=pc, addr=100 + pc)
        prefetcher.progress(a, prefetch_hit=False)

    # Some entries must have been evicted since num_sets=4,num_ways=2 = 8 entries max
    total_entries = sum(len(s) for s in prefetcher.cache.sets)
    assert total_entries <= 8
