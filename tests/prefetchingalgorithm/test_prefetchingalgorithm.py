from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm


class DummyAlgo(PrefetchAlgorithm):
    def init(self):
        pass

    def progress(self, address, prefetch_hit):
        return [address + 1]

    def close(self):
        pass


def test_dummy_algo_progress():
    algo = DummyAlgo()
    algo.init()
    preds = algo.progress(10)
    assert preds == [11]
    algo.close()
