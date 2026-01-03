from collections import defaultdict

from compact.cache.replacementpolicy.ReplacementPolicy import ReplacementPolicy


class LfuReplacementPolicy(ReplacementPolicy):
    def __init__(self):
        self.freq = defaultdict(int)  # frequency counters per key

    def touch(self, key: int):
        # normal access (hit) → increase frequency
        self.freq[key] += 1

    def evict(self) -> int:
        # return the least frequently used key
        if not self.freq:
            return None
        victim = min(self.freq, key=lambda k: self.freq[k])
        return victim

    def insert(self, key: int):
        # insert with initial frequency 1
        self.freq[key] = 1

    def remove(self, key: int):
        if key in self.freq:
            del self.freq[key]

    def prefetch_hit(self, key: int):
        # on a prefetch hit → optional boost
        self.freq[key] += 2  # give higher weight to prefetch success
