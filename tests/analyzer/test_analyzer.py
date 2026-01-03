from compact.analyzer.analyzer import Analyzer
from compact.dataloader.impl.ArrayDataLoader import ArrayLoader
from compact.prefetchingalgorithm.impl.referencepredictiontable import RptAlgorithm
from compact.prefetchingalgorithm.memoryaccess import MemoryAccess


def test_analyzer_counts_correct_and_incorrect(capfd):
    data = [
        MemoryAccess(address=0, pc=0),
        MemoryAccess(address=4, pc=0),
        MemoryAccess(address=8, pc=0),
        MemoryAccess(address=12, pc=0),
        MemoryAccess(address=16, pc=0),
    ]
    loader = ArrayLoader(data=data)
    rpt = RptAlgorithm()
    analyzer = Analyzer(rpt, loader)
    analyzer.run()
