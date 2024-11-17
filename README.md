# Custom Question-Answering Model with PyTorch

This repository contains the implementation of a custom transformer-based Question-Answering (QA) system inspired by BERT, built using PyTorch and trained on the Stanford Question Answering Dataset (SQuAD).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Training Setup](#training-setup)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Work](#future-work)

---

## Overview

This project involves building a custom QA model capable of extracting answers from a given context for a specific question. The architecture closely mirrors BERT, with added flexibility to train efficiently on a subset of the SQuAD dataset.

---

## Architecture

The model architecture consists of:

1. **Embedding Layer**:

   - Word embeddings
   - Positional embeddings
   - Token type embeddings

2. **Transformer Layers**:

   - Multi-head self-attention mechanism to extract contextual relationships.
   - Feed-forward network for intermediate representations.

3. **QA-Specific Output Heads**:
   - Separate linear layers to predict the start and end token positions for the answer span.

---

## Dataset

The [SQuAD dataset](https://huggingface.co/datasets/squad) was used for training and validation.

- **Training Subset**: 10,000 samples.
- **Validation Subset**: 1,000 samples.

The dataset was tokenized using the Hugging Face `BertTokenizerFast`, with context truncation to 512 tokens.

---

## Training Setup

- **Hardware**: Dual NVIDIA Tesla T4 GPUs.
- **Batch Size**: 8.
- **Epochs**: 5.
- **Optimizer**: AdamW.
- **Mixed-Precision Training**: Used AMP for faster training without sacrificing accuracy.
- **Parallelization**: Enabled multi-GPU training with `torch.nn.DataParallel`.

---

## Results

The model demonstrated the ability to correctly predict answers on validation examples:

- **Question**: What is the capital of France?

  - **Context**: France, located in Western Europe, has Paris as its capital and largest city.
  - **Prediction**: _Paris_

- **Question**: Who wrote the novel '1984'?
  - **Context**: '1984' is a dystopian social science fiction novel and cautionary tale, written by the English writer George Orwell in 1949.
  - **Prediction**: _George_

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/lee4real/qa-bert-pytorch.git
cd qa-bert-pytorch
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Train the Model

To train the model on the SQuAD dataset:

```bash
python train.py
```

### 4. Test the Model

You can test the model on a sample question and context:

```bash
python predict.py --question "What is the capital of France?" --context "France, located in Western Europe, has Paris as its capital and largest city."
```

---

## Future Work

- Fine-tuning on domain-specific datasets for better performance in specialized tasks.
- Model deployment as an API for real-time question answering.
- Exploring knowledge distillation to reduce model size while maintaining accuracy.

---

## Acknowledgments

- [Hugging Face Datasets ](https://huggingface.co/datasets/squad) for providing SQuAD.
- PyTorch for enabling custom transformer implementation.

---

## License

This project is licensed under the MIT License.
