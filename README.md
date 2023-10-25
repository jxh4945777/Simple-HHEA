# Simple-HHEA
A simple but effective method (and two new proposed datasets: ICEWS-WIKI, ICEWS-YAGO) for entity alignment on highly heterogeneous knowledge graphs.

- [Finished] Dataset
- [Finished] Source Code

The rise of knowledge graph (KG) applications underscores the importance of entity alignment (EA) across diverse KGs. While existing EA datasets provide some insights, their limited heterogeneity often falls short in capturing the complexities of practical KGs. These practical KGs, with their varying scales, structures, and sparse overlapping entities, present a unique set of challenges for EA.

In light of these challenges, this repository introduces **Simple-HHEA**, a method tailored for aligning highly heterogeneous KGs (HHKGs).

## Highlights

- **Addressing Dataset Limitations**: We introduce two new HHKG datasets that are designed to emulate real-world EA scenarios, thereby bridging the gap between experimental and practical settings.
  
- **Extensive Evaluation**: Our extensive experiments on these datasets reveal that conventional message-passing and aggregation mechanisms often struggle to leverage valuable structure information in HHKGs. This observation is especially true for existing GNN-based EA methods.

- **Introducing Simple-HHEA**: In response to these findings, we present Simple-HHEA, an innovative approach that adeptly integrates entity name, structure, and temporal information. Unlike many complex methods, Simple-HHEA offers a straightforward yet powerful solution for navigating the intricacies of HHKGs.

- **Future-Oriented Insights**: Our experiments suggest that the future of EA model design should prioritize adaptability and efficiency. Models should be able to handle varying information quality and effectively discern patterns across HHKGs.

## Getting Started

1. **Dependencies**: Ensure you have the required libraries installed, including `numpy`, `pandas`, `torch`, `scipy`.
2. **Execution**: Launch the main script with the desired parameters:
    ```bash
    python main_Simple_HHEA_bi_info.py --lang 'icews_wiki' --name_noise_ratio 0.0 --structure_noise_ratio 0.0 --if_structure True
    ```

## Resources

The datasets and comprehensive source code for this project can be found [here](https://anonymous.4open.science/r/HHEA/).

## Contribution & Feedback

We welcome any contributions to the codebase and dataset. If you find any issues or have suggestions, please raise them in the GitHub repository. Your feedback is invaluable in refining and expanding Simple-HHEA.
