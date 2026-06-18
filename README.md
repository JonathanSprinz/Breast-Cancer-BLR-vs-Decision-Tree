**LSE Data Analytics Career Accelerator | DA301 — Advanced Analytics for Organisational Impact**

## Overview
This project builds and compares two classification models to predict whether a breast 
cancer case is malignant or benign — Binary Logistic Regression (BLR) and a pruned 
Classification Decision Tree. Both models are evaluated on the same dataset and 
train/test split for a fair comparison.

## Business Problem
Westside Hospital wants to predict breast cancer diagnoses from detection measurements. 
The model classifies each case as:
- **M = 1** → Malignant (cancer)
- **B = 0** → Benign (healthy)

## Dataset
- **File:** breast_cancer_data.csv
- **Rows:** 569
- **Columns:** 32 (31 independent variables + 1 dependent variable: diagnosis)
- **Class distribution:** Benign = 357, Malignant = 212
- No missing values, no string encoding required, no SMOTE needed

## Results

| Model | Training Accuracy | Test Accuracy |
|---|---|---|
| Binary Logistic Regression | — | 95.91% |
| Decision Tree (unpruned) | 100.0% ← overfitting | 91.23% |
| Decision Tree (pruned) | — | 95.91% |

Both the BLR and pruned decision tree achieve **95.91% accuracy** on unseen test data.

## Methodology

### Binary Logistic Regression
- Encoded diagnosis as binary (M=1, B=0)
- 70/30 train/test split (random_state=0)
- Fitted with sklearn LogisticRegression (max_iter=10000)
- Evaluated with confusion matrix and classification report

### Classification Decision Tree
- Unpruned tree — 100% training accuracy, 91.23% test accuracy (overfitting)
- Pruned tree — Gini criterion, max_depth=5, min_samples_leaf=5, min_samples_split=5
- Pruning corrected overfitting and matched BLR accuracy at 95.91%
- Final model trained on full dataset and visualised with tree.plot_tree()

## Key Findings
- **Overfitting is clearly demonstrated** — the unpruned tree memorises training data 
  (100% accuracy) but generalises poorly (91.23% test). Pruning resolves this.
- **Both models achieve identical accuracy** — neither is superior on this dataset.
- **Interpretability advantage** — the decision tree plot shows exactly which features 
  drive each split, valuable for explaining decisions to clinical stakeholders.
- **In medical contexts, recall matters more than accuracy** — a false negative 
  (missing a cancer case) is far more costly than a false positive.
- **95.91% significantly outperforms** the Telcom churn model (79%), reflecting 
  cleaner and more predictive features in this dataset.

## Libraries Used
```python
pandas, numpy, sklearn, statsmodels, seaborn, matplotlib
```

## How to Run
1. Clone the repo
2. Place `breast_cancer_data.csv` in the same folder as the notebook
3. Open `Breast_Cancer_Classification_Decision_Tree.ipynb` and run all cells in order
4. No additional library installs required

## Skills Demonstrated
- Binary Logistic Regression (BLR)
- Classification Decision Tree
- Decision tree pruning (Gini, max_depth, min_samples)
- Overfitting identification and correction
- Model comparison on identical train/test splits
- Confusion matrix and classification report interpretation
- Recall vs accuracy trade-off in medical classification contexts
- Python: pandas, sklearn, seaborn, matplotlib
