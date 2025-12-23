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


# References

The following prefetching algorithms are implemented in COMPACT. The references for these algorithms are provided in the `paper.bib` file.

- [@bestoffset]
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