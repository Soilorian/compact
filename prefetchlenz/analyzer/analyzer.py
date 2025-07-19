import logging

from prefetchlenz.dataloader.dataloader import DataLoader
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.analysis")


class Analyzer:
    """
    Runs a prefetch algorithm on a data stream and tallies predictions.

    Counts correct vs. incorrect prefetches.
    """

    def __init__(self, algorithm: PrefetchAlgorithm, dataloader: DataLoader):
        """
        Args:
            algorithm (PrefetchAlgorithm): Prefetch algorithm instance.
            dataloader (DataLoader): Source of address stream.
        """
        self.algorithm = algorithm
        self.dataloader = dataloader

    def run(self):
        """Execute the simulation and print results."""
        self.algorithm.init()
        correct = 0
        incorrect = 0
        pending = set()

        self.dataloader.load()
        dataSize = len(self.dataloader)
        logger.info(f"Starting analysis on {dataSize} data")

        for idx in range(dataSize):
            addr = self.dataloader[idx]
            preds = self.algorithm.progress(addr)
            for p in preds:
                pending.add(p)

            if idx + 1 < dataSize:
                next_addr = self.dataloader[idx + 1]
                if next_addr in pending:
                    correct += 1
                    pending.remove(next_addr)
                    logger.debug(f"Correct: {next_addr}")
                else:
                    incorrect += 1
                    logger.debug(f"Incorrect: {next_addr}")

        self.algorithm.close()
        logger.info(f"Correct predictions: {correct}")
        logger.info(f"Incorrect predictions: {incorrect}")
