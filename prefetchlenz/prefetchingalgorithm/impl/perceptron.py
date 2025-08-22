import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.PerceptronPrefetcher")


@dataclass
class PerceptronEntry:
    address: int
    pc: int
    features: Dict[str, Any]
    depth: int = 0
    stride: int = 0
    valid: bool = False


class PerceptronPrefetcher(PrefetchAlgorithm):
    def __init__(
        self, cache_line_size_bytes=64, page_size_bytes=4096, max_stride_history=16
    ):
        """
        Initializes the Perceptron Prefetcher.

        Args:
            cache_line_size_bytes (int): Size of a cache line in bytes.
            page_size_bytes (int): Size of a page in bytes.
            max_stride_history (int): The number of strides to track for the underlying prefetcher.
        """
        self.cache_line_size_bytes = cache_line_size_bytes
        self.page_size_bytes = page_size_bytes
        self.max_stride_history = max_stride_history

        # Perceptron state
        self.weight_tables = {
            "pc": {},
            "address": {},
            "depth": {},
            "pc_xor_depth": {},
            "pc_xor_stride": {},
        }
        self.prefetch_table: Dict[int, PerceptronEntry] = {}
        self.reject_table: Dict[int, PerceptronEntry] = {}
        self.pc_last_access = {}
        self.last_address_per_pc = {}

        # PPF thresholds, based on common microarchitectural designs
        self.TAU_LO = -8  # Lower threshold for rejection
        self.TAU_HI = (
            8  # Upper threshold for L2 placement (not fully implemented in this sim)
        )
        self.WEIGHT_MAX = 15
        self.WEIGHT_MIN = -16
        self.TRAIN_THRESHOLD_POSITIVE = 1
        self.TRAIN_THRESHOLD_NEGATIVE = -1

    def init(self):
        """Initialize state before simulation begins."""
        self.weight_tables = {
            "pc": {},
            "address": {},
            "depth": {},
            "pc_xor_depth": {},
            "pc_xor_stride": {},
        }
        self.prefetch_table = {}
        self.reject_table = {}
        self.pc_last_access = {}
        self.last_address_per_pc = {}
        logger.info("PerceptronPrefetcher initialized.")

    def _get_features(
        self, access: MemoryAccess, depth: int, stride: int
    ) -> Dict[str, Any]:
        """
        Extracts features from the memory access to be used by the perceptron.
        This is a simplified set of features based on the paper's suggestions.
        """
        page_addr = access.address // self.page_size_bytes
        cache_line_addr = access.address // self.cache_line_size_bytes

        # Using a simple hash for PC XOR depth and PC XOR stride
        # In a real implementation, a more robust hash would be used.
        pc_xor_depth_hash = access.pc ^ depth
        pc_xor_stride_hash = access.pc ^ stride

        return {
            "pc": access.pc,
            "address": cache_line_addr,
            "depth": depth,
            "page_address": page_addr,
            "stride": stride,
            "pc_xor_depth": pc_xor_depth_hash,
            "pc_xor_stride": pc_xor_stride_hash,
        }

    def _predict(self, features: Dict[str, Any]) -> int:
        """
        Makes a prediction by summing the weights of the features.
        """
        perceptron_sum = 0
        for feature_name, value in features.items():
            if feature_name in self.weight_tables:
                weight = self.weight_tables[feature_name].get(value, 0)
                perceptron_sum += weight
        return perceptron_sum

    def _train(self, features: Dict[str, Any], useful: bool):
        """
        Updates the perceptron weights based on the outcome.
        """
        perceptron_sum = self._predict(features)

        # Update weights if prediction was wrong or weak, to avoid over-training
        # as described in the paper[cite: 945, 946].
        update_condition = False
        if useful:
            # Positive outcome (useful prefetch): update if prediction was weak or negative
            if perceptron_sum < self.TRAIN_THRESHOLD_POSITIVE:
                update_condition = True
        else:
            # Negative outcome (useless prefetch): update if prediction was weak or positive
            if perceptron_sum > self.TRAIN_THRESHOLD_NEGATIVE:
                update_condition = True

        if update_condition:
            direction = 1 if useful else -1
            for feature_name, value in features.items():
                if feature_name in self.weight_tables:
                    current_weight = self.weight_tables[feature_name].get(value, 0)
                    new_weight = current_weight + direction
                    # Saturating counters, as described in the paper [cite: 921]
                    new_weight = min(self.WEIGHT_MAX, max(self.WEIGHT_MIN, new_weight))
                    self.weight_tables[feature_name][value] = new_weight

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Processes a single memory access, performs training, and makes prefetch predictions.
        """
        # --- TRAINING PHASE ---
        # The paper describes training on demand accesses or evictions.
        # Here, we will train on a demand access if it was a prefetch hit or a miss.
        # We need to check if the current access address exists in our tables for training.

        # Check Prefetch Table for a hit.
        # The 'prefetch_hit' flag indicates the address was requested and was in cache due to prefetching.
        if prefetch_hit and access.address in self.prefetch_table:
            entry = self.prefetch_table.pop(access.address)
            self._train(entry.features, useful=True)

        # Check Reject Table for a false negative.
        # This occurs if we previously rejected a prefetch but it was then accessed later.
        if access.address in self.reject_table:
            entry = self.reject_table.pop(access.address)
            self._train(entry.features, useful=True)

        # --- PREDICTION PHASE ---
        # PPF is a filter, so we need to generate prefetch candidates from an underlying prefetcher first.
        # Here, we use a simple stride prefetcher.
        prefetch_candidates = []
        if access.pc in self.last_address_per_pc:
            last_access = self.last_address_per_pc[access.pc]
            current_stride = access.address - last_access

            # Simple stride prefetching: predict next N blocks
            # We will generate up to 4 candidates to simulate deep speculation.
            for i in range(1, 5):
                candidate_address = access.address + (current_stride * i)
                prefetch_candidates.append(candidate_address)

        # Update last access for the stride prefetcher
        self.last_address_per_pc[access.pc] = access.address

        # --- FILTERING PHASE ---
        final_prefetches = []
        for depth, candidate in enumerate(prefetch_candidates, start=1):
            features = self._get_features(access, depth, current_stride)
            confidence_sum = self._predict(features)

            if confidence_sum > self.TAU_LO:
                # Prefetch is accepted. Store it in the Prefetch Table for later training.
                final_prefetches.append(candidate)
                # This is a simplified table, using only the address and features.
                # The paper's tables also store a tag.
                self.prefetch_table[candidate] = PerceptronEntry(
                    address=candidate,
                    pc=access.pc,
                    features=features,
                    depth=depth,
                    stride=current_stride,
                    valid=True,
                )
            else:
                # Prefetch is rejected. Store it in the Reject Table for later training.
                self.reject_table[candidate] = PerceptronEntry(
                    address=candidate,
                    pc=access.pc,
                    features=features,
                    depth=depth,
                    stride=current_stride,
                    valid=True,
                )

        return final_prefetches

    def close(self):
        """Clean up state after simulation ends."""
        logger.info("PerceptronPrefetcher closed.")
