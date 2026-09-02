# Adaptive Hierarchical Cue Hashing for Cross-modal Retrieval (AHCH)

AHCH is a supervised cross-modal hashing framework that combines cue learning, adaptive hierarchical fusion (AHF), and semantic instance alignment (SIA) for image-text retrieval.


## Framework

![Overview of the AHCH framework](./main.jpg)

## Repository Files

| File | Description |
| --- | --- |
| `config.py` | Experiment configuration file. |
| `train.py` | Training code and entry point. |
| `load_data.py` | Dataset loading and splitting code. |
| `log.txt` | Training log supplied with the repository. |
| `main.jpg` | Framework overview image. |

Additional paths mentioned below are proposed examples; their inclusion in the repository has not been verified.

## Computing Environment

The supplied screenshot shows the following machine configuration. This records the available environment; it does not establish which experiments were executed on it.

| Component | Configuration |
| --- | --- |
| Operating system | Ubuntu 22.04 |
| Python | 3.12 |
| PyTorch | 2.5.1 |
| CUDA shown in the environment image | 12.4 |
| GPU | 1 × NVIDIA GeForce RTX 4090, 24 GB VRAM |
| CPU allocation | 22 vCPUs on an AMD EPYC 7T83 64-Core Processor host |
| System memory | 90 GB |
| System disk | 30 GB |
| Data disk | 50 GB |


## Datasets and Preparation

Download the original datasets from their respective sources:

1. [MIRFLICKR25K](https://press.liacs.nl/mirflickr/)
2. [NUS-WIDE](https://github.com/NExTplusplus/NUS-WIDE)
3. [MS COCO](https://cocodataset.org/)

Original datasets remain subject to their providers' access conditions and licenses.

The supplied manuscript describes the following protocol for each source dataset:

| Subset | Sampling rule |
| --- | --- |
| Query set | 5,000 randomly selected image-text pairs. |
| Known retrieval database, D_k | All remaining pairs from the source dataset. |
| Training set | 10,000 pairs sampled from D_k. |
| Unknown retrieval database, D_unk | Combined samples from the other two datasets. |

The training and query sets are disjoint; the training set is a subset of D_k. Images are resized to 224 × 224 pixels, and text is tokenized using Byte Pair Encoding (BPE).


## Training Configuration


```python
dataset = "MIRFLICKR25K"
data_root = "./data/MIRFLICKR25K"
backbone = "CLIP ViT-B/32"
freeze_backbone = True
fusion_variant = "LF"
hash_length = 32
batch_size = 64
optimizer = "Adam"
learning_rate = 1e-4
weight_decay = 1e-5
epochs = 100
seed = 42
alpha = 0.5
beta = 0.5
lambda_quantization = 5.0
output_dir = "./outputs/MIRFLICKR25K/32bits/seed42"
```

Evaluate hash lengths of 16, 32, and 64 bits as described in the manuscript. Loss weights and other settings must be recorded separately for each dataset.


```bash
python train.py
```


## Evaluation Protocol

Evaluate image-to-text (I2T) and text-to-image (T2I) retrieval separately. The manuscript reports mAP, precision-recall curves, Top-N precision, and NDCG@1000. Report D_k and D_unk results separately and identify the dataset, code length, and retrieval direction for every result.



```bash
python train.py --mode evaluate \
  --checkpoint ./outputs/MIRFLICKR25K/32bits/seed42/final.pt
```


## Implementation Details to Verify

They are not claims about the current implementation.

| Topic | Illustrative specification requiring verification |
| --- | --- |
| Cue construction | Generate class cues using training information only. For an example deterministic multi-label rule, choose the lowest positive class index; this is a convention, not a definition of a dominant category. |
| Inference cues | Select cues using the available query or gallery input without its ground-truth labels. The actual selector must be specified and evaluated. |
| Label isolation | Use query and gallery labels only for evaluation relevance, outside the encoding path. Training-subset labels remain available for supervised training. |
| Fusion variants | Use LF for the example main configuration; evaluate WF as a separately named variant. Parameter matching requires measurement. |
| Loss normalization | Use B for paired mini-batch size, N for total training pairs, C for categories, and K for hash length. State the exact reduction for each loss. |
| Binarization | Example design: sign in the forward pass and an identity straight-through gradient for a binary training path. Use a separate detached binary target for a squared quantization penalty. Verify that this matches the implemented gradient paths and manuscript equations. |

Adopting any example that differs from the implementation may require method changes and renewed evaluation.


