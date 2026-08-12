# Scientific Abstract Classification with BERT

A text classification project that fine-tunes **BERT** to automatically classify scientific paper abstracts into three research fields:

- Bioinformatics
- Environmental Science
- Neuroscience

## Project Overview

This project was developed for the **Artificial Intelligence (G1)** course.

Scientific abstracts were collected from **PubMed** and used to build a balanced dataset. A pre-trained `bert-base-uncased` model was then fine-tuned for three-class text classification.

The model was evaluated using **5-fold Stratified K-Fold cross-validation** and a completely isolated test set.

## Dataset

The dataset was collected using the PubMed API.

| Field | Number of Abstracts |
|---|---:|
| Bioinformatics | 359 |
| Neuroscience | 364 |
| Environmental Science | 380 |
| **Total** | **1,103** |

Abstracts shorter than 100 characters, duplicate papers, and records without both titles and abstracts were removed.

## Model

- **Base model:** `bert-base-uncased`
- **Task:** 3-class sequence classification
- **Max sequence length:** 256
- **Batch size:** 32
- **Learning rate:** 2e-5
- **Epochs:** 2
- **Optimizer:** AdamW
- **Loss function:** CrossEntropyLoss
- **Cross-validation:** 5-fold Stratified K-Fold

The full BERT model was fine-tuned end-to-end together with the classification head.

## Results

### 5-Fold Cross-Validation

- Mean Accuracy: **92.05%**
- Mean Precision: **92.16%**
- Mean Recall: **92.00%**
- Mean F1-score: **92.02%**

### Final Test Set

The final model was evaluated on an isolated test set of 220 abstracts.

- Accuracy: **90.00%**
- Macro Precision: **90.21%**
- Macro Recall: **89.96%**
- Macro F1-score: **90.03%**

Neuroscience achieved the highest class-level F1-score at **92.20%**.

## Tools & Libraries

- Python
- PyTorch
- Hugging Face Transformers
- BERT
- PubMed / NCBI E-utilities
- scikit-learn
- pandas

## Summary

This project demonstrates how a fine-tuned BERT model can be used to automatically classify scientific abstracts with strong performance. The final model achieved **90% accuracy** on unseen test data while maintaining stable performance across cross-validation folds.

Future improvements could include larger datasets, longer-sequence models, ensemble methods, multi-label classification, and domain adaptation.
