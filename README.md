# Reuse-Aware Graph Learning for Efficient Skeleton-Based Action Recognition

Undergraduate research project (OUR). Student: Abhiraj Agarwal; mentor: Arnab Ganguly.

## The idea

Skeleton-based action recognition (SBAR) treats an action as a time-ordered sequence of body graphs: joints are
vertices, bones are edges. Graph convolutional networks such as ST-GCN [1] and STTGC-Net [2] (which adds spatial and
temporal self-attention) do well on the NTU RGB+D benchmarks [5]. But they compute every neighbourhood of every sample
from scratch, even when a very similar neighbourhood has already been processed.

**Hypothesis:** similar actions contain recurring local graph structures, and unrelated actions share fewer of them.
Different dance forms share local arm and leg movements; a dance move and a cricket shot do not. If the network has
already learned a local neighbourhood, a sufficiently similar neighbourhood in another sample may not need to be
computed again. Skeleton graphs become large once time is included, so this reuse could cut training cost at a
small, measurable cost in accuracy.

**What is new:** existing SBAR work mostly improves accuracy (richer graphs, attention, multiple streams) or makes
individual computations cheaper [3]. Computation reuse in general GNNs, such as HAG [4] or reusing historical node
representations across iterations, is exact (identical aggregations within one graph) or temporal. This project
studies *approximate* reuse *across samples*: a k-hop neighbourhood is reused when it is similar enough to one
already computed. The research question is not only whether two neighbourhoods look alike, but whether they are alike
enough to reuse the computation with a bounded error in the embedding. Measuring similarity, compute saved and
embedding error together gives an explicit accuracy-compute trade-off.

## Plan

1. **Baseline and redundancy profiling.** Reproduce an ST-GCN / STTGC-Net-style model on NTU RGB+D 60 and measure
   parameters, FLOPs, training time and GPU memory. Then measure how often local k-hop spatio-temporal neighbourhood
   signatures recur across samples and action classes.
2. **Subgraph extraction and similarity.** Pick salient joints ("action points") with a simple motion signal such
   as per-joint motion energy [3], and extract their k-hop spatio-temporal neighbourhoods. Summarise topology with
   Weisfeiler-Lehman refinement (joint / body-part identities as initial labels), and node / edge attributes
   (positions, velocities, bone vectors) with permutation-invariant sketches such as weighted MinHash. Test whether
   similar actions produce more similar signatures than unrelated ones.
3. **Reuse-aware training and aggregation.** Keep a cache of learned subgraph embeddings or partial layer outputs.
   A neighbourhood similar enough to a cached entry (threshold τ) is reused or warm-started; otherwise it is
   computed and added to the cache. Subgraph summaries are pooled or combined with attention into the clip
   representation. Sweeping τ gives the accuracy-compute trade-off. The comparison against the full-computation
   baseline covers accuracy, FLOPs, training time, memory, cache hit rate, accuracy on confusable actions, stale
   cache entries and sketching overhead.

```
skeleton sequence -> seed joints ("action points", -> k-hop spatio-temporal -> signatures: WL labels (topology)
(joints x frames)    e.g. by motion energy)            subgraphs                 + weighted MinHash (attributes)
                  -> LSH buckets + reuse cache (threshold τ) -> pooled graph embedding -> classifier
```

**Datasets.** Primary: NTU RGB+D 60 (56,880 clips, 60 actions) and NTU RGB+D 120 (114,480 clips, 120 actions) [5],
3D skeletons with a fixed 25-joint layout. Supplementary: AIST++ (1,408 dance sequences, 10 genres) and YNU-Dance for
the dance hypothesis; MSRC-12, MSR Action3D and UTKinect-Action3D for quick sanity checks.

## Where the code is now

Stage 1, the baseline. Despite the repository's name, the model code is **MST-GCN** (Chen et al., AAAI 2021), an
ST-GCN-family model. It is not the full STTGC-Net: the STTA self-attention block is not implemented. Results from it
are reported as an MST-GCN baseline. On NTU-60 with 300-frame clips it has 2.9 M parameters and costs
17.3 GMACs per clip. Its 10 blocks widen from 112 to 448 channels, and by the last block one output position sees the
whole clip.

The baseline run is set up to feed stage 1's redundancy profiling. After training, `tools/export_run.py` saves, for
every train and val clip, the embedding after each of the 10 blocks together with labels, predictions and logits.
It also writes a report with a per-block analysis: output shape, cost, receptive field, how well each block's
embedding separates the actions, and how similar clips of the same and of different actions are at each depth.
These are the full-computation reference numbers the reuse experiments will be compared against.

Not implemented yet: the STTA attention block, NTU-120 data preparation, the k-hop subgraph signatures (WL, MinHash,
LSH) and the reuse cache.

## Running it

- **Windows GPU workstation (the training machine):** [WORKSTATION.md](WORKSTATION.md). Three scripts:
  `scripts\setup_env.ps1` (Python, PyTorch with CUDA, packages), `scripts\prepare_ntu.ps1` (dataset), then
  `scripts\run_training.ps1 -Config config\ntu\ntu60_xsub_joint.yaml` for a complete run that resumes after
  interruptions and ends with the results in `results\ntu60_xsub_joint\report.md`.
- **Anywhere else / details of the training code and what was changed from the original:** [SETUP.md](SETUP.md).

| path | contents |
|---|---|
| `main.py` | training / evaluation loop (resumable checkpoints, progress lines, memory limits) |
| `model/` | MST-GCN: `ST_GCN.py` (model), `layers.py` (multi-scale graph and temporal convolutions) |
| `graph/` | the NTU 25-joint skeleton graph |
| `feeders/` | dataset loader (memory-mapped `.npy`) |
| `data_gen/` | raw NTU `.skeleton` files -> normalised `.npy` (`ntu_gendata.py`) |
| `config/` | run configs: `ntu/ntu60_{xsub,xview}_joint.yaml`, `smoke_test.yaml` |
| `tools/` | `export_run.py` (results + embeddings), `inspect_data.py` (data / model inspection), `bench_train.py` (speed and memory), `make_dummy_data.py` |
| `scripts/` | Windows PowerShell scripts for the workstation (setup, data, run, monitoring) |
| `resource_limits.py` | caps on GPU memory / RAM (and optionally CPU / GPU time) for a shared machine |

## References

1. Yan, S., Xiong, Y., Lin, D. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.
   AAAI 2018. https://doi.org/10.1609/aaai.v32i1.12328
2. Yagneshwar, T., Mukherjee, S. STTGC-Net: Spatial-Temporal Transformer with Graph Convolution for Skeleton-Based
   Action Recognition. ICVGIP 2023. https://doi.org/10.1145/3627631.3627640
3. Sun, S., Cheng, Z., Zhang, Z., Dong, J., Li, Z., Wang, M. Exploring Adaptive Masked Reconstruction for
   Self-Supervised Skeleton-Based Action Recognition. CVPR 2026, 13974-13983.
4. Jia, Z., Lin, S., Ying, R., You, J., Leskovec, J., Aiken, A. Redundancy-Free Computation for Graph Neural
   Networks. KDD 2020. https://doi.org/10.1145/3394486.3403142
5. Shahroudy, A., Liu, J., Ng, T.-T., Wang, G. NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis.
   CVPR 2016. https://doi.org/10.1109/CVPR.2016.115

Baseline model: Chen, Z., Li, S., Yang, B., Li, Q., Liu, H. Multi-Scale Spatial Temporal Graph Convolutional Network
for Skeleton-Based Action Recognition. AAAI 2021.
