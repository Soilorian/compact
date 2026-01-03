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
    name: Sharif University of Technology
  - index: 2
    name: IPM Institute For Research In Fundamental Sciences
date: 3 November 2025
bibliography: paper.bib
---

# Summary

COMPACT (Comprehensive Prefetching Algorithm Framework) is a Python-based simulation framework designed to support the rapid development and evaluation of hardware prefetching algorithms. It offers a modular and extensible environment that allows researchers and engineers to explore and test new prefetching techniques without requiring complex hardware changes. The framework includes several common prefetching algorithms, a flexible cache simulator, and built-in tools for analyzing the prefetcher performance. COMPACT is designed for ease of use and customization, making it a practical and valuable resource for the computer architecture research community.

# Statement of Need

As the performance gap between processors and main memory continues to grow, hardware prefetching has become a key approach to improving overall system performance. However, developing and evaluating new prefetching algorithms often requires significant time and resources. COMPACT helps address this issue by providing a lightweight and adaptable simulation framework that allows researchers to efficiently test and compare prefetching strategies. By simplifying the evaluation process, COMPACT supports faster experimentation and innovation in the study of hardware prefetching.


# Implemented Algorithms

The following prefetching algorithms are implemented in COMPACT. Brief descriptions are provided for each algorithm based on their original publications. The references for these algorithms are provided in the `paper.bib` file.

- **Best-Offset Prefetcher** [@bestoffset]: Learns the optimal offset value for sequential prefetching by testing a predefined list of offsets. During a training phase, it evaluates each offset to detect recurring patterns (X, X+d) in recent memory accesses. Once training completes, the prefetcher uses the best-scoring offset to generate prefetches, dynamically adapting to varying program behaviors and access patterns.
- **B-Fetch Prefetcher** [@bfetch]: Uses branch prediction information to direct data prefetching by maintaining a Branch Stream Table that maps branch program counters to stream descriptors. By leveraging branch prediction signals, it prefetches data along predicted instruction paths ahead of execution, reducing cache misses for branch-directed memory accesses.
- **Correlation Prefetcher** [@correlation]: Learns address correlations from the memory access stream by tracking patterns where accessing address A is often followed by address B. When correlations are detected with sufficient confidence, it issues prefetches for the predicted addresses, effectively handling irregular access patterns through correlation-based prediction.
- **Domino Prefetcher** [@domino]: A temporal data prefetcher that predicts future memory addresses based on miss sequences using two miss history tables. MHT1 tracks single-miss patterns while MHT2 tracks two-miss patterns, allowing the prefetcher to generate predictions by correlating recent miss sequences with historical patterns observed in the access stream.
- **EBCP Prefetcher** [@ebcp]: An epoch-based correlation prefetcher that partitions the access stream into epochs defined by quiescence periods. It correlates the first miss of each epoch with the miss sequences of subsequent epochs, learning temporal relationships between epoch boundaries to prefetch data ahead of epoch transitions.
- **Event-Triggered Prefetcher** [@eventtriggered]: An event-driven programmable prefetcher designed for irregular workloads. It uses configurable rules based on exponential weighted moving averages of metrics like stride, latency, and inter-arrival time to trigger prefetches, allowing customization for different workload characteristics.
- **F-TDC Prefetcher** [@f_tdc_prefetcher]: Efficient footprint caching for tagless DRAM caches that tracks per-page reference patterns using compact footprint vectors. By maintaining historical reference information embedded in page table entries, it prefetches chunks within pages based on previously observed access patterns.
- **Feedback-Directed Prefetcher** [@feedbackdirected]: Improves prefetcher performance and bandwidth efficiency by using feedback metrics to dynamically adjust prefetch aggressiveness. It monitors accuracy, lateness, and pollution metrics over sampling intervals and adaptively tunes prefetch parameters to optimize cache performance.
- **GHB Prefetcher** [@ghb]: Uses a global history buffer to correlate memory access patterns across different program counters. By maintaining a circular buffer of all accesses linked through predecessor relationships, it identifies recurring address sequences and generates prefetches based on delta correlations learned from the global access history.
- **Graph Prefetcher** [@graph]: Exploits data structure knowledge for graph workloads by tracking memory access patterns that correspond to graph traversal operations. It identifies graph edges and node relationships from the access stream and prefetches connected nodes ahead of traversal, optimized for irregular graph data structures.
- **HDS Prefetcher** [@hds]: A dynamic hot data stream prefetcher that identifies frequently accessed data sequences through bursty profiling and builds deterministic finite state machines. It dynamically learns hot streams during profiling phases and uses these learned patterns to prefetch data during execution phases.
- **IPCP Prefetcher** [@ipcp]: Instruction pointer classifier-based spatial prefetching that classifies program counters into constant stride, spatial signature, or complex patterns. It employs specialized sub-prefetchers for each classification, adapting its prefetching strategy based on the memory access pattern exhibited by each instruction pointer.
- **LearnCluster Prefetcher** [@learncluster]: Combines k-means clustering with LSTM-based prediction to learn memory access patterns. It partitions the address space into clusters, builds per-cluster delta vocabularies, and uses a shared LSTM encoder to predict top-K address deltas within each cluster for prefetching.
- **LearnPrefetch Prefetcher** [@learnprefetch]: Uses LSTM neural networks to learn memory access patterns from address delta sequences. By training on historical miss sequences, it predicts future address deltas and generates prefetch candidates, leveraging machine learning to capture complex temporal dependencies in memory access behavior.
- **Linear Prefetcher** [@linear]: Linearizes irregular memory accesses to enable improved correlated prefetching. It transforms non-sequential access patterns into stream-like sequences using hardware-like stream buffers, making irregular accesses amenable to standard prefetching techniques.
- **Markov Predictor Prefetcher** [@markovpredictor]: Employs Markov chain predictors to model memory access sequences with variable-order dependencies. It maintains transition probabilities between address states and generates prefetches based on the most likely successor addresses given recent access history.
- **Metadata Prefetcher** [@metadata]: Efficient metadata management for irregular data prefetching that maintains per-PC correlation tables mapping trigger addresses to predicted targets. It tracks confidence counters for each correlation and issues prefetches when correlations exceed a threshold, optimizing storage through compact metadata representation.
- **Neural Prefetcher** [@neural]: A hierarchical neural model for data prefetching that combines global and local prediction components. It uses a global LSTM to capture long-term patterns and per-PC linear models for localized predictions, creating a two-level hierarchy that adapts to both global and instruction-specific access patterns.
- **Perceptron Prefetcher** [@perceptron]: Perceptron-based prefetch filtering that uses a linear classifier to decide which prefetch candidates to issue. It trains perceptron weights based on features extracted from the access stream and filters out low-confidence prefetches, improving prefetch accuracy by reducing ineffective prefetches.
- **SMS Prefetcher** [@sms]: Spatial memory streaming that tracks per-region access patterns using an accumulation table and pattern history table. It builds spatial signatures for memory regions and prefetches blocks within regions based on previously observed access patterns, optimized for spatial locality.
- **Store-Ordered Streamer Prefetcher** [@storeorderedstreamer]: Records per-producer store streams and uses consumer accesses to trigger prefetches along producer memory access sequences. When a consumer accesses a line that appears in a producer's stream, it prefetches subsequent lines from that producer's store sequence.
- **TCP Prefetcher** [@tcp]: Tag correlating prefetcher that operates on cache line tags rather than full addresses. It learns correlations between consecutive line tags and predicts successor tags, supporting prediction chaining for multi-level prefetching and using per-PC context for improved accuracy.
- **Temporal Memory Streaming Prefetcher** [@temporalmemorystreaming]: Records temporal memory access sequences in a circular miss order buffer and replays these sequences when matching addresses are encountered. It maintains a directory mapping addresses to sequence positions and prefetches subsequent addresses from recorded sequences.
- **Triage Prefetcher** [@triage]: Temporal prefetching without off-chip metadata that learns address neighbor relationships through PC-localized training. It maintains an on-chip metadata cache with adaptive sizing based on prefetch effectiveness, tracking temporal correlations between consecutive addresses per program counter.
- **Triangel Prefetcher** [@triangel]: A high-performance on-chip temporal prefetcher that combines pattern-based and neighbor-based prediction. It uses a training unit to discover address patterns and maintains Markov metadata to predict future accesses, optimizing for both accuracy and timeliness in temporal prefetching.

# References

- [@bfetch]
- [@bingo]
- [@correlation]
- [@domino]
- [@dspatch]
- [@ebcp]
- [@eventtriggered]
- [@f_tdc_prefetcher]
- [@feedbackdirected]
- [@ghb]
- [@graph]
- [@hds]
- [@indirectmemory]
- [@ipcp]
- [@learncluster]
- [@learnprefetch]
- [@linear]
- [@markovpredictor]
- [@metadata]
- [@neural]
- [@perceptron]
- [@sms]
- [@storeorderedstreamer]
- [@tcp]
- [@tempo]
- [@temporalmemorystreaming]
- [@triage]
- [@triangel]

[^1]: These authors contributed equally to this work.