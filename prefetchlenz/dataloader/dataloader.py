import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger("prefetchLenz.dataloader")


class DataLoader(ABC):
    """Interface for data loaders that supply address streams."""

    @abstractmethod
    def load(self):
        """Return a sequence of memory addresses."""
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __iter__(self):
        """Return an iterator for this class"""
        pass

    @abstractmethod
    def __len__(self):
        pass


class ArrayLoader(DataLoader):
    """Loads addresses from a Python list."""

    def __init__(self, data: List[int]):
        """
        Args:
            data (List[int]): Pre-collected address sequence.
        """
        self.data = data

    def load(self):
        """Return the array of addresses."""
        logger.debug(f"ArrayLoader loading {len(self.data)} addresses")

    def __getitem__(self, item):
        return self.data[item]

    @abstractmethod
    def __iter__(self):
        return self.data.__iter__()

    @abstractmethod
    def __len__(self):
        return self.data.__len__()
