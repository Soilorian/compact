from abc import ABC, abstractmethod


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
