# STMGraph

## Overview
 ![Image text](https://github.com/binbin-coder/SpatialG/blob/main/overview.jpg)
    Spatial transcriptomics (ST) technologies enable dissecting the tissue architecture in spatial context. To perceive the global contextual information of gene expression patterns in tissue, the spatial dependence of cells must be fully considered by integrating both local and non-local features by means of spatial-context-aware. However, the current ST integration algorithm ignoring ST dropouts, hindering the spatial-aware of ST features, resulting in challenges in the accuracy and robustness of microenvironmental heterogeneity detecting, spatial domain clustering, and batch-effects correction. Here, we developed an STMGraph, a universal dual-view dynamic deep learning framework that combines dual-remask (MASK-REMASK) with dynamic graph attention model (DGAT) to exploit ST data outperforming pre-existing tools. The dual-remask mechanism masks the embeddings before encoding and decoding, establishing dual-decoding-view to share features mutually. DGAT leverages self-supervision to update graph linkage relationships from two distinct perspectives, thereby generating a comprehensive representation for each node. Systematic benchmarking against with ten state-of-the-art tools, reveals that the STMGraph has the best performance with high accuracy and robustness on spatial domain clustering for the datasets of diverse ST platforms from multi- to sub-cellular resolutions. Furthermore, STMGraph aggregates ST information cross regions by dual-remask to realize the batch-effects correction implicitly, allowing for spatial domain clustering of ST multi-slices. STMGraph is platform independent, and superior in spatial-context-aware to achieve microenvironmental heterogeneity detection, spatial domain clustering, batch-effects correction, and new biological discovery, therefore, a desirable novel tool for diverse ST studies.

## Software dependencies
scanpy

pytorch

pyG

## Tutorials
see https://github.com/binbin-coder/STMGraph

