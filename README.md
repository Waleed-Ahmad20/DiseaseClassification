# Disease Category Classification

A machine learning project that classifies diseases into broader medical categories using clinical feature data. The project compares K-Nearest Neighbors (KNN) and Logistic Regression models across two feature encoding strategies: TF-IDF and One-Hot encoding.

## Overview

The dataset covers 25 diseases spanning five medical categories:

| Category | Diseases |
|---|---|
| **Cardiovascular** | Acute Coronary Syndrome, Aortic Dissection, Atrial Fibrillation, Cardiomyopathy, Heart Failure, Hyperlipidemia, Hypertension |
| **Respiratory/Infectious** | Asthma, COPD, Pneumonia, Pulmonary Embolism, Tuberculosis |
| **Neurological/Endocrine** | Alzheimer, Epilepsy, Migraine, Multiple Sclerosis, Pituitary Disease, Stroke |
| **Endocrine** | Adrenal Insufficiency, Diabetes, Thyroid Disease |
| **Gastrointestinal** | Gastritis, Gastro-oesophageal Reflux Disease, Peptic Ulcer Disease, Upper Gastrointestinal Bleeding |

Each disease is described by four clinical feature types: **Risk Factors**, **Symptoms**, **Signs**, and **Subtypes**.

## Repository Structure

```
├── 22I2647_BSE8B_AssNo3.ipynb            # Main Jupyter notebook (data prep, modelling, evaluation)
├── disease_features.csv                  # Raw dataset with textual clinical features per disease
├── encoded_output2.csv                   # One-hot encoded feature matrix
├── disease_category_mapping_generated.csv # Disease-to-category mapping produced by the notebook
├── model_evaluation_results_categories.csv# Cross-validation results for all model configurations
├── disease_category_classifier_app.py    # Streamlit dashboard for exploring model results
└── Task4.pdf                             # Assignment specification
```

## Methodology

### Feature Engineering

- **TF-IDF**: Each feature type (Risk Factors, Symptoms, Signs, Subtypes) is vectorised independently using a bigram TF-IDF vectoriser, then the four matrices are concatenated and L2-normalised.
- **One-Hot**: A pre-built binary matrix where each column represents the presence or absence of a specific clinical feature.

Dimensionality reduction is applied before modelling (TruncatedSVD / PCA, 17 components) to reduce noise and improve generalisation.

### Models

| Model | Variants |
|---|---|
| **KNN** | k ∈ {3, 5, 7}; distance metrics: Euclidean, Manhattan, Cosine |
| **Logistic Regression** | Default solver with L2 regularisation |

### Evaluation

All models are evaluated with **5-fold stratified cross-validation**, restricted to disease categories that have at least five samples. Metrics reported: Accuracy, Precision (weighted), Recall (weighted), F1-score (weighted).

### Results (Top Configurations)

| Encoding | Model | k | Metric | Avg Accuracy | Avg F1-score |
|---|---|---|---|---|---|
| TF-IDF | KNN | 5 | Cosine | 0.8000 | 0.7556 |
| TF-IDF | KNN | 5 | Euclidean | 0.7833 | 0.7528 |
| TF-IDF | Logistic Regression | — | — | 0.7667 | 0.7056 |
| TF-IDF | KNN | 3 | Cosine | 0.7667 | 0.7022 |

TF-IDF consistently outperforms One-Hot encoding. KNN with k=5 and cosine distance achieves the best overall accuracy (80%) and F1-score.

## Setup & Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy streamlit
```

### Run the Notebook

Open and run all cells in `22I2647_BSE8B_AssNo3.ipynb` using Jupyter. The notebook will:
1. Load and preprocess the feature data.
2. Generate TF-IDF and One-Hot feature matrices.
3. Train and evaluate KNN and Logistic Regression models.
4. Save results to `model_evaluation_results_categories.csv`.

### Launch the Streamlit Dashboard

```bash
streamlit run disease_category_classifier_app.py
```

The dashboard allows interactive exploration of model results by selecting the model type, feature encoding, k value, and distance metric from the sidebar.

## Dataset

`disease_features.csv` contains one row per disease with the following columns:

- `Disease` – disease name
- `Risk Factors` – list of known risk factors
- `Symptoms` – list of reported symptoms
- `Signs` – list of observable clinical signs
- `Subtypes` – list of recognised disease subtypes
