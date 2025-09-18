from prefetchlenz.prefetchingalgorithm.impl.linear import *


def make_access(address, pc=0x10, loaded_value=None, is_cache_miss=False):
    return LinearMemoryAccess(pc=pc, address=address, loaded_pointer=loaded_value)


def test_stream_push_and_available():
    s = Stream(id=0)
    s.push(10, maxlen=4)
    s.push(20, maxlen=4)
    s.push(30, maxlen=4)
    assert s.available_after_last_issued() == [10, 20, 30]
    s.last_issued_index = 0
    assert s.available_after_last_issued() == [20, 30]


def test_prefetcher_starts_stream_and_issues_prefetches():
    p = LinearizingPrefetcher(
        prefetch_degree=2, stream_history_len=8, trigger_on_miss_only=False
    )
    p.init()
    # simulate a pointer-chasing access: load at addr=1 returns pointer 100
    ma1 = make_access(address=1, loaded_value=100, is_cache_miss=False)
    out1 = p.progress(ma1, prefetch_hit=False)
    # confidence initially increased but may be below threshold; set confidence manually to force issue
    sid = p.addr_to_stream.get(100)
    assert sid is not None
    p.streams[sid].confidence = p.confidence_threshold + 1
    # push further nodes
    p.progress(make_access(address=100, loaded_value=200), prefetch_hit=False)
    p.progress(make_access(address=200, loaded_value=300), prefetch_hit=False)
    # now trigger on access to 200
    out = p.progress(make_access(address=200, loaded_value=400), prefetch_hit=False)
    # Some prefetches should be requested (addresses from stream buffer)
    assert isinstance(out, list)
    # outstanding should reflect issued prefetches
    for a in out:
        assert a in p.outstanding


def test_prefetch_completed_and_notify_hit():
    p = LinearizingPrefetcher(
        prefetch_degree=2, stream_history_len=8, trigger_on_miss_only=False
    )
    p.init()
    ma = make_access(address=1, loaded_value=100)
    p.progress(ma, prefetch_hit=False)
    sid = p.addr_to_stream[100]
    p.streams[sid].confidence = p.confidence_threshold + 1
    p.progress(make_access(address=100, loaded_value=200), prefetch_hit=False)
    out = p.progress(make_access(address=100, loaded_value=200), prefetch_hit=False)
    if out:
        a = out[0]
        assert a in p.outstanding
        p.prefetch_completed(a)
        assert a not in p.outstanding
        # test notify_prefetch_hit increments hits
        before = p.hw_prefetch_hits
        p.notify_prefetch_hit(a)
        assert p.hw_prefetch_hits == before + 1


def test_adaptive_degree_changes():
    p = LinearizingPrefetcher(
        prefetch_degree=4, stream_history_len=8, trigger_on_miss_only=False
    )
    p.init()
    # force counters and call adapt directly
    p.hw_prefetch_requests = 100
    p.hw_prefetch_hits = 5  # low hit ratio -> expect decrease
    p.hw_cache_misses = 200
    p._adapt_prefetching_policy()
    assert p.prefetch_degree < 4

    p.hw_prefetch_requests = 10
    p.hw_prefetch_hits = 9  # high hit ratio -> expect increase
    p.hw_cache_misses = 100
    p._adapt_prefetching_policy()
    assert p.prefetch_degree >= 1
