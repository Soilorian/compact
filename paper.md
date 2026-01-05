---
title: 'COMPACT: Comprehensive Prefetching Algorithm Framework'
authors:
  - name: Mahdi Alinejad[^1]
    affiliation: '1'
  - name: Sahand Zoufan[^1]
    affiliation: '1'
  - name: Atiyeh Gheibi-Fetrat
    affiliation: '1'
  - name: Shaahin Hessabi
    affiliation: '1'
  - name: Hamid Sarbazi-Azad
    affiliation: '1,2'
affiliations:
  - index: 1
    name: Sharif University of Technology, Iran
  - index: 2
    name: IPM Institute For Research In Fundamental Sciences, Iran
date: 3 November 2025
bibliography: paper.bib
---

# Summary

COMPACT (Comprehensive Prefetching Algorithm Framework) is a Python-based simulation framework designed to support the rapid development and evaluation of hardware prefetching algorithms. It offers a modular and extensible environment that allows researchers and engineers to explore and test new prefetching techniques without requiring complex hardware changes. The framework includes several common prefetching algorithms, a flexible cache simulator, and built-in tools for analyzing the prefetcher performance. COMPACT is designed for ease of use and customization, making it a practical and valuable resource for the computer architecture research community.

# Statement of Need

As the performance gap between processors and main memory continues to grow, hardware prefetching has become a key approach to improving overall system performance. However, developing and evaluating new prefetching algorithms often requires significant time and resources. COMPACT helps address this issue by providing a lightweight and adaptable simulation framework that allows researchers to efficiently test and compare prefetching strategies. By simplifying the evaluation process, COMPACT supports faster experimentation and innovation in the study of hardware prefetching.


# Implemented Algorithms

The following prefetching algorithms are implemented in COMPACT. Brief descriptions are provided for each algorithm based on their original publications.

- **Markov Predictor** [@markovpredictor]: Represents addresses as states in a Markov chain and learns transition probabilities between states from historical access patterns. During execution, access to a state triggers prediction of the most likely successor states, and the corresponding addresses are prefetched.
- **Correlation Prefetching** [@correlation]: Uses a helper memory thread to observe sequences of cache misses and learn correlations between them. When a similar miss pattern is detected, it replays the correlated miss sequence to prefetch data ahead of demand.
- **Dynamic Hot Data Stream Prefetching** [@hds]: Identifies frequently accessed memory streams by profiling bursty access patterns and models these hot streams using deterministic finite state machines (DFSMs). During profiling, frequent access sequences are encoded in DFSMs, and at runtime the prefetcher follows DFSM states to predict and prefetch future accesses.
- **TCP** [@tcp]: Correlates cache line tags rather than full memory addresses. It maintains a history of recently accessed cache line tags and predicts the next tag based on these correlations. By chaining predicted tags, it can prefetch subsequent cache lines ahead of the demand access.
- **Global History Buffer** [@ghb]: Maintains a global history buffer of recent memory accesses and correlates address deltas across program counters. It predicts future addresses by detecting recurring delta patterns and issues prefetches accordingly.
- **Store-Ordered Streaming** [@storeorderedstreamer]: Records per-producer store streams and uses consumer accesses to trigger prefetches along producer memory access sequences. When a consumer accesses a line that appears in a producer's stream, it prefetches subsequent lines from that producer's store sequence.
- **Spatial Memory Streaming** [@sms]: Tracks per-region access patterns using an accumulation table and pattern history table. It builds spatial signatures for memory regions and prefetches blocks within regions based on previously observed access patterns, optimized for spatial locality.
- **Epoch-Based Correlation Prefetching** [@ebcp]: Partitions the access stream into epochs whose boundaries are defined by quiescence periods. It learns correlations between the first miss of an epoch and subsequent misses within that epoch, and, upon detecting a similar epoch trigger, replays the learned miss sequence to prefetch ahead of demand.
- **Feedback-Directed Prefetching** [@feedbackdirected]: Monitors feedback metrics such as prefetch accuracy, lateness, and cache pollution, and adjusts prefetch parameters (e.g., prefetch degree) over sampling intervals based on these metrics.
- **Temporal Instruction Fetch Streaming** [@temporalmemorystreaming]: Records instruction access sequences in a circular miss-order buffer. When an instruction is fetched, it checks the buffer for a matching address and, upon a match, prefetches subsequent instructions from the recorded sequence. A directory maps instruction addresses to sequence positions to support efficient sequence replay.
- **Linearizing Irregular Memory Accesses** [@linear]: Transforms irregular memory accesses into stream-like sequences by logically linearizing the access patterns. This mapping makes non-sequential accesses appear sequential, enabling conventional prefetching mechanisms to predict and prefetch the next address in the transformed stream.
- **B-Fetch** [@bfetch]: Uses branch prediction information to direct data prefetching by maintaining a Branch Stream Table that maps branch program counters to stream descriptors. By leveraging branch prediction signals, it prefetches data along predicted instruction paths ahead of execution, reducing cache misses for branch-directed memory accesses.
- **IMP** [@indirectmemory]: Targets indirect access patterns in which future addresses depend on previous load values. It learns correlations between index values and their corresponding target addresses and uses the learned index-to-target relationship to predict and prefetch future targets.
- **Best-Offset Hardware Prefetching** [@bestoffset]: Evaluates a predefined set of candidate address offsets and maintains a usefulness score for each candidate. Based on periodic evaluation, it selects the highest-scoring offset to generate prefetches and re-evaluates offsets to track changes in access behavior.
- **Efficient Footprint Caching** [@f_tdc_prefetcher]: Learns page-level spatial access footprints using compact footprint vectors and predicts future accesses within a page by fetching cache blocks associated with a previously accessed page. This approach is particularly relevant for tagless DRAM cache designs.
- **Graph Prefetching** [@graph]: Uses programmer- or compiler-provided graph traversal semantics to guide prefetching. It identifies access patterns associated with traversal and prefetches connected nodes ahead of traversal.
- **Translation-Triggered Prefetching** [@tempo]: Uses address translation activity as a trigger for prefetching by combining physical page mappings with cache-line offsets to predict future memory accesses. Prefetches are issued non-speculatively as part of the translation process.
- **Domino Temporal Data Prefetcher** [@domino]: Uses two history tables to capture first-order and second-order correlations between cache misses. Predictions from these tables are chained to generate prefetches that reflect deeper temporal relationships in the miss stream.
- **Event-Triggered Programmable Prefetcher** [@eventtriggered]: A programmable prefetcher that issues prefetches only when specified events are triggered based on runtime memory access characteristics (e.g., stride, latency, and timing). Event conditions are defined by the programmer or system and can be used to adapt prefetching behavior to workload characteristics.
- **Learning Memory Access Patterns** [@hashemi2018]: Uses an LSTM model to learn temporal dependencies from address delta sequences derived from cache misses. By training on historical miss sequences, it predicts future address deltas and generates prefetch candidates.
- **Bingo Spatial Data Prefetcher** [@bingo]: Tracks access patterns within regions using bit-vector footprints and associates these footprints with trigger events. When a trigger matches a stored footprint, it issues prefetches for the cache blocks within the region indicated by the matched footprint.
- **DSPatch** [@dspatch]: Maintains accuracy-biased and coverage-biased footprints for each spatial signature and selects between them based on observed behavior. Prefetches are issued by aligning the selected footprint with the region's trigger offset.
- **Efficient Metadata Management** [@metadata]: Maintains per-program-counter metadata that correlates trigger addresses with predicted target addresses and associates confidence counters with each correlation. Prefetches are issued when confidence exceeds a threshold to reduce ineffective prefetches.
- **Perceptron-Based Prefetch Filtering** [@perceptron]: Filters prefetch candidates using a perceptron classifier that evaluates features from the memory access stream. The perceptron is trained online using feedback from prior prefetch decisions, reinforcing high-confidence prefetches and suppressing low-confidence ones.
- **Temporal Prefetching Without the Off-Chip Metadata** [@triage]: Tracks temporal correlations between consecutive addresses using on-chip metadata. The metadata cache is adaptively sized based on observed prefetch effectiveness, and prefetches are issued when high-confidence temporal correlations are identified.
- **Bouquet of Instruction Pointers** [@ipcp]: Instruction pointer classifier-based spatial prefetching that classifies program counters into constant stride, spatial signature, or complex patterns. It employs specialized sub-prefetchers for each classification, adapting its prefetching strategy based on the memory access pattern exhibited by each instruction pointer.
- **Hierarchical Neural Model of Data Prefetching** [@neural]: Combines a global LSTM model for capturing long-range access patterns with per-program-counter linear models for localized instruction-specific behavior. Prefetches are generated from the combined global and localized predictions.
- **Triangel** [@triangel]: Combines pattern-based prediction with neighbor-based prediction by learning recurring access patterns and maintaining Markov-style temporal metadata for neighboring address relationships. Prefetches are triggered using predictions derived from both mechanisms.

# References

- [@markovpredictor]
- [@correlation]
- [@hds]
- [@tcp]
- [@ghb]
- [@storeorderedstreamer]
- [@sms]
- [@ebcp]
- [@feedbackdirected]
- [@temporalmemorystreaming]
- [@linear]
- [@bfetch]
- [@indirectmemory]
- [@bestoffset]
- [@f_tdc_prefetcher]
- [@graph]
- [@tempo]
- [@domino]
- [@eventtriggered]
- [@hashemi2018]
- [@bingo]
- [@dspatch]
- [@metadata]
- [@perceptron]
- [@triage]
- [@ipcp]
- [@neural]
- [@triangel]

[^1]: These authors contributed equally to this work.