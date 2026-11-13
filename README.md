# Credit Card Fraud Detection

A machine learning project that identifies potentially fraudulent credit card transactions using classification algorithms. The project focuses on handling highly imbalanced data, comparing different models, and optimizing prediction thresholds to improve fraud detection.

## Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a very small fraction of all transactions.

A model that simply predicts every transaction as legitimate can achieve high accuracy while failing to detect actual fraud. To address this, this project explores different classification techniques and evaluates them using metrics that are more meaningful for imbalanced datasets.

## Dataset

**Source:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains transactions made by European cardholders over two days.

| Attribute | Description |
|---|---|
| Time | Seconds elapsed between each transaction and the first transaction |
| Amount | Transaction amount |
| V1–V28 | Anonymized features obtained using PCA |
| Class | Target variable (0 = Legitimate, 1 = Fraudulent) |

The dataset is highly imbalanced, with fraudulent transactions accounting for approximately 0.17% of the observations in the original dataset.

Download `creditcard.csv` and place it inside the `data/` directory.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Imbalanced Learning:** Imbalanced-learn (SMOTE)
- **Model Persistence:** Joblib
- **Environment:** Jupyter Notebook

## Machine Learning Models

The following algorithms are implemented and compared:

1. Logistic Regression
2. Random Forest Classifier
3. HistGradientBoosting Classifier

The models are trained and evaluated using a stratified train-validation-test split.

Class weighting is used in supported models to account for the imbalance between legitimate and fraudulent transactions.

An additional experiment explores SMOTE to understand its impact on model performance.

## Project Workflow

### 1. Exploratory Data Analysis
- Examined dataset structure, distributions and statistical properties.
- Analyzed the distribution of legitimate and fraudulent transactions.
- Studied transaction amounts and their relationship with fraud.

### 2. Data Preprocessing
- Checked and handled duplicate records.
- Separated input features and target labels.
- Applied feature scaling to numerical columns.
- Used stratified sampling to preserve class proportions.

### 3. Model Training
- Trained multiple classification algorithms.
- Used class weighting to address class imbalance.
- Compared model performance on validation data.

### 4. Model Evaluation
Evaluated the models using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Precision-Recall AUC (PR-AUC)
- Confusion Matrix

PR-AUC and Recall receive particular attention because failing to identify fraudulent transactions can be costly.

### 5. Threshold Optimization
Instead of relying on the default classification threshold of 0.5, the decision threshold is optimized using validation data to maximize the F1-score.

The selected threshold is then used for the final evaluation on the untouched test set.

### 6. Model Interpretation
- Analyzed feature importance using Random Forest.
- Identified features that contribute most to model predictions.
- Compared the performance of different classification approaches.

## Project Structure

```text
credit_card_fraud_detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── fraud_detection_model.joblib
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
│
├── reports/
│   ├── validation_model_comparison.csv
│   └── final_test_metrics.csv
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- Jupyter Notebook or JupyterLab

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
```

Navigate to the project directory:

```bash
cd credit-card-fraud-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Dataset Setup

Download the dataset from Kaggle and place `creditcard.csv` inside the `data/` directory.

### Run the Notebook

```bash
jupyter lab
```

Open:

```text
notebooks/credit_card_fraud_detection.ipynb
```

Execute the notebook cells sequentially.

## Results

After executing the notebook, the project generates:

- Model comparison results on validation data.
- Precision-recall curves.
- Classification reports.
- Confusion matrix.
- Feature importance visualization.
- Final test evaluation metrics.
- A saved model with its optimized decision threshold.

Actual performance metrics are generated after training and evaluation.

## Key Learnings

- Handling highly imbalanced classification datasets.
- Understanding the limitations of accuracy as an evaluation metric.
- Comparing linear and ensemble-based classification algorithms.
- Applying class weighting and SMOTE.
- Understanding the precision-recall trade-off.
- Preventing data leakage during preprocessing and model evaluation.
- Optimizing classification thresholds.
- Interpreting machine learning predictions using feature importance.

---
