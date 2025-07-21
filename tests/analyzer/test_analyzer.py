from prefetchlenz.analyzer.analyzer import Analyzer
from prefetchlenz.dataloader.dataloader import ArrayLoader
from prefetchlenz.prefetchingalgorithm.impl.referencepredictiontable import RptAlgorithm


def test_analyzer_counts_correct_and_incorrect(capfd):
    data = [0, 4, 8, 12, 16]
    loader = ArrayLoader(data)
    rpt = RptAlgorithm()
    analyzer = Analyzer(rpt, loader)
    analyzer.run()
    out, err = capfd.readouterr()

    assert "Correct Predictions: 3" in out
    assert "Incorrect Predictions: 1" in out
