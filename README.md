# LoRA from Scratch: Efficient RoBERTa Fine-Tuning

This project implements Low-Rank Adaptation (LoRA) from scratch to fine-tune a RoBERTa-base model on specific GLUE benchmark tasks. We focused on replicating the original LoRA paper's results while introducing specific engineering optimizations to improve training efficiency and resource utilization on standard hardware.

## Team
* **Arda Ağaçdelen**
* **Yiğit Kaya Bağcı**

## Project Overview
We applied LoRA to the **RoBERTa** architecture to demonstrate efficient model tuning. By freezing the pre-trained weights and injecting trainable rank decomposition matrices, we significantly reduced the number of trainable parameters while maintaining performance.

* **Total Parameters:** ~125,000,000
* **Trainable Parameters (LoRA):** 887,042
* **Reduction:** We tuned only **~0.7%** of the total parameters.

## Implementation & Methodology

### Datasets
We evaluated performance on the **GLUE Benchmark**, specifically:
1.  **SST-2** (Stanford Sentiment Treebank)
2.  **CoLA** (Corpus of Linguistic Acceptability)

### Model Configuration
* **Target Modules:** Query and Value weights
* **Rank:** 8
* **Alpha:** 16
* **Epochs:** 60
* **Loss Function:** Cross Entropy Loss

### Our Contributions & Optimizations
We also tried modified training pipeline to maximize GPU efficiency on our hardware (NVIDIA T4).

| Feature | Paper / Standard | **Our Implementation** | **Impact** |
| :--- | :--- | :--- | :--- |
| **Batch Size** | 16 | **256** | Utilized GPU VRAM more efficiently. |
| **Precision** | FP32 | **TF32** | Activated Ampere-specific hardware acceleration. |
| **Optimizer** | AdamW | **Fused AdamW** | Eliminated CPU bottlenecks during updates. |

**Result:** The tuning process was approximately **10x faster** with these contributions.

## Results

We compared our LoRA implementation (with and without our optimizations) against the pre-trained baseline and the original paper's reported results.

### SST-2 Accuracy
| Method | Accuracy (%) |
| :--- | :--- |
| **Our LoRA (Optimized)** | **94.70** |
| Our LoRA (Unoptimized) | 94.80 |
| Paper LoRA Result | ~95.1 |
| Pretrained Baseline | ~83.60 |

### CoLA Accuracy
| Method | Accuracy (%) |
| :--- | :--- |
| **Our LoRA (Optimized)** | **63.10** |
| Our LoRA (Unoptimized) | 65.80 |
| Paper LoRA Result | ~63.4 |
| Pretrained Baseline | ~37.23 |

## Environment & Tech Stack

The project was executed in **Google Colab** with the following specifications:

* **GPU:** NVIDIA Tesla T4 (16 GB GDDR6)
* **CUDA:** Version 12.2
* **Language:** Python 3.10
* **Dependency Resolver:** uv (used for fast dependency management)
* **Libraries:**
    * PyTorch, NumPy
    * Transformers, Datasets, Safetensors
* **Libraries:**