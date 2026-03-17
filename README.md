# Extractive Question Answering on SQuAD 2.0 (Text Mining and Data Visualization)

A comparative study of extractive QA models — from a  BiLSTM-BiDAF baseline to a fine-tuned DeBERTa-v3 augmented with RAG — evaluated with both automatic metrics and LLM-based human-like assessment.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models](#models)
  - [BiLSTM + BiDAF](#1-bilstm--bidaf)
  - [DeBERTa-v3 + RAG + LLM-Based evaluation](#2-deberta-v3--rag)
- [Interactive Dashboard](#interactive-dashboard)
- [Repository Structure](#repository-structure)
- [Reproducibility Instructions](#reproducibility-instructions)

---

## Introduction

This project tackles the **Extractive Question Answering** task using the Stanford Question Answering Dataset 2.0 (SQuAD 2.0). The pipeline covers the full NLP lifecycle: dataset sampling and preprocessing, rich exploratory analysis, model training and fine-tuning, LLM-powered evaluation, and an interactive dashboard for result visualization.

The project compares two fundamentally different approaches to QA, then uses a LLaMA model accessed via API to audit whether EM and F1 scores faithfully capture model quality on unanswerable questions and paraphrastic answers.

---

## Dataset

**Source:** [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) — Stanford Question Answering Dataset (Rajpurkar et al., 2018)

SQuAD 2.0 extends SQuAD 1.1 by introducing **unanswerable questions**: questions that look plausible but have no answer in the provided passage. This forces models to understand when to abstain rather than always produce a span.

### Sampling Strategy

The full SQuAD 2.0 training set was sampled to obtain a working subset of approximately **90,000 examples**.

Each example contains:
- `context` — a Wikipedia passage
- `question` — a natural language question
- `answers` — a list of answer spans (empty list if unanswerable)
- `is_impossible` — boolean flag for unanswerability
---

## Exploratory Data Analysis

Notebook: [`01_eda.ipynb`](01_eda.ipynb)

A thorough EDA was conducted before any modelling. Key analyses include:

**Token-level analysis**
- Distribution of context lengths, question lengths, and answer span lengths in tokens
- Token overlap between question and context
- Most common tokens and bigrams

**Linguistic analysis**
- Most common question starters (*who*, *what*, *when*, *where*, *why*, *how*)
- **Named Entity Recognition (NER)** — entity type distribution in answers (PER, ORG, LOC, …); analysis of which entity types are most commonly the target of questions
- **Part-of-Speech (POS) tagging** — distribution of POS tags in questions and answers
- Dependency parsing

**Structural patterns**
- Most common topic analysis
- Unanswerable vs answerable question length comparison
- Vocabulary size and token frequency distributions

All EDA plots are reproduced interactively in the dashboard (see [Interactive Dashboard](#interactive-dashboard)).

---

## Models

### 1. BiLSTM + BiDAF

Notebook: [`02_bilstm_bidaf.ipynb`](02_bilstm_bidaf.ipynb)

The first model is a reimplementation of the **Bidirectional Attention Flow (BiDAF)** architecture (Seo et al., 2017), enriched with bidirectional LSTMs at each encoding layer.

**Architecture overview:**

```
Character Embedding Layer   →  char-level CNN
Word Embedding Layer        →  GloVe 300d
Phrase Embedding Layer      →  BiLSTM
Attention Flow Layer        →  Context-to-Query & Query-to-Context attention
Modelling Layer             →  BiLSTM × 2
Output Layer                →  Linear(start), Linear(end)
```

The attention flow layer computes a similarity matrix between every context token and every query token, then derives both C2Q (context-to-query) and Q2C (query-to-context) attention in a single step. This allows the model to condition its span prediction on a rich, query-aware context representation without losing the full bidirectional signal.

**Key design choices:**
- GloVe pre-trained embeddings (frozen during training)
- Character-level CNN for handling OOV tokens
- Cross-entropy loss on start and end positions independently
- Trained from scratch without any pre-trained transformer backbone

---

### 2. DeBERTa-v3 + RAG + LLM-based Evaluation 

Notebook: [`03_deberta_rag.ipynb`](03_deberta_rag.ipynb)

The second model fine-tunes **`microsoft/deberta-v3-base`** on SQuAD 2.0 with a Retrieval-Augmented Generation (RAG) extension.

**Why DeBERTa-v3?**
DeBERTa (Decoding-enhanced BERT with disentangled attention) improves over BERT and RoBERTa by:
- Using **disentangled attention** — content and position embeddings are kept separate and attended over independently
- Adding a **virtual token** at the output layer to improve span boundary detection
- DeBERTa-v3 further replaces MLM pre-training with **ELECTRA-style replaced token detection**, yielding stronger representations with the same compute

**RAG extension:**
A lightweight retrieval component was added on top of the standard extractive QA head. For each question, a small set of candidate passages is retrieved from a document index (FAISS) and re-ranked before the reader processes the top-k results. This simulates an open-domain QA setup layered on top of the closed-domain SQuAD evaluation.

**Fine-tuning setup:**
- Model: `microsoft/deberta-v3-base`
- Tokenizer: fast tokenizer with sliding window for long contexts (stride = 128, max length = 512)
- Optimizer: AdamW with linear warmup + linear decay
- Hardware: GPU (RTX5090 depending on availability)
- Handling of unanswerable questions: the model is trained to predict a null span `[CLS]` when `is_impossible = True`

**LLM-Based evaluation (LLaMa)**
Standard QA metrics (EM and F1) are known to be imperfect: they penalise semantically correct paraphrases and may reward lucky surface overlaps. To audit this, a **LLaMA** model was queried via API to act as an independent judge.

**Evaluation protocol:**
For a random sample of model predictions, LLaMA was given:
- The context passage
- The question
- The gold answer
- The predicted answer from each model

It was then asked to rate the prediction on a 1–5 scale and provide a brief justification. The LLM scores were compared against EM and F1 to measure correlation and identify systematic failure modes (e.g., cases where F1 = 0 but the answer is semantically correct).

---

**Key takeaways:**
- DeBERTa-v3 outperforms BiDAF by a large margin across all metrics, confirming the strength of large pre-trained transformers even on classical extractive QA benchmarks
- The RAG component doesn't provide an improvement on questions probabibly beacuse the number of context analyzed is too low.
- LLM-based evaluation broadly agrees with F1 rankings but reveals cases where automatic metrics are misleading, especially around unanswerable questions

---

## Interactive Dashboard

Notebook: [`04_dashboard.ipynb`](04_dashboard.ipynb)

A fully interactive dashboard built with **Plotly** and **Dash** that brings together all results in one place.

**Sections:**

- **EDA** — interactive versions of all exploratory plots
- **Model Comparison** — side-by-side EM and F1 bar charts, LLaMA score distributions, confusion matrices for answerability classification
- **Live QA Demo** — input a custom question and context, run inference on both models, and compare their predictions in real time
- **Metric Audit** — scatter plot of F1 vs LLaMA score per example, with filters to isolate disagreement cases

Run the dashboard locally with:

```bash
jupyter notebook 05_dashboard.ipynb
# or, if exported as a standalone app:
python dashboard_app.py
```

---

## Repository Structure

```
.
├── 01_eda.ipynb                  # Exploratory Data Analysis
├── 02_bilstm_bidaf.ipynb         # BiLSTM + BiDAF model: training & evaluation
├── 03_deberta_rag_llm.ipynb      # DeBERTa-v3 fine-tuning + RAG extension + LLaMA-based evaluation of automatic metrics
├── 05_dashboard.ipynb            # Interactive Plotly dashboard
│
├── data/
│   ├── train_sampled.json   # Sampled training set (~70k examples)
│   └── val_sampled.json     # Validation set
|   └── test_sampled.json    # Test set
│
├── models/
│   ├── bidaf/                    # BiDAF model weights and config
│   └── deberta/                  # DeBERTa-v3 fine-tuned checkpoint
│
├── results/
│   ├── bidaf_predictions.json
│   ├── deberta_predictions.json
│   └── llm_scores.json           # LLaMA evaluation scores
│
├── utils/
│   ├── preprocessing.py          # Tokenization, sliding window, data loaders
│   ├── metrics.py                # EM, F1, normalization utilities
│   └── rag.py                    # FAISS index + retrieval helpers
│
├── requirements.txt
└── README.md
```

---

> **GPU note:** for DeBERTa-v3 fine-tuning a GPU with at least 16 GB VRAM is recommended. For inference only, 8 GB is sufficient. The BiDAF model can be trained on CPU, though training is significantly faster on GPU.

---

## Reproducibility Intructions

To reproduce the experiments, please follow the steps below:

**1. Clone the repository**

git clone [https://github.com/ElisaQuadrini/TextMiningDataVis_Project.git](https://github.com/ElisaQuadrini/TextMiningDataVis_Project.git) 

cd TextMiningDataVis_Project


**2. Download the pre-trained/fine-tuned weights on your pc:**
   [Google Drive Folder](https://drive.google.com/drive/folders/1qrieC-mHhRIrC2OecUTZwMuXtSQfHDb3)
   

**3. Ensure the model path is correctly set before execution.**


**4. Open and run the notebook:**

   01_EDA.ipynb

   02_BiLSTM_BiDAF
   
   03_DeBERTav3_RAG_LLM.ipynb



To use the Dashboard:
---

## Citation

If you use this work or build on it, please cite the original dataset and model papers:

```bibtex
@inproceedings{rajpurkar2018squad2,
  title     = {Know What You Don't Know: Unanswerable Questions for SQuAD},
  author    = {Rajpurkar, Pranav and Jia, Robin and Liang, Percy},
  booktitle = {ACL},
  year      = {2018}
}

@inproceedings{he2021deberta,
  title  = {DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
  author = {He, Pengcheng and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu},
  booktitle = {ICLR},
  year   = {2021}
}

@inproceedings{seo2017bidaf,
  title  = {Bidirectional Attention Flow for Machine Comprehension},
  author = {Seo, Minjoon and Kembhavi, Aniruddha and Farhadi, Ali and Hajishirzi, Hannaneh},
  booktitle = {ICLR},
  year   = {2017}
}
```

---

