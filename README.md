# Credit Card Fraud Detection – Anomaly Detection with Z‑Score

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange) ![Colab](https://img.shields.io/badge/Google%20Colab-supported-yellow)

## 📌 Overview

This project applies a **simple rule‑based anomaly detection** method to the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The dataset contains 284,807 transactions, of which only **492 (0.17%) are fraudulent** – a highly imbalanced real‑world scenario.

Instead of complex machine learning, we use the **Z‑score** on the transaction `Amount` feature: any transaction whose amount is more than 3 standard deviations away from the mean is flagged as a potential fraud.

> **Key learning:** This project demonstrates that even a simple univariate rule can catch some fraudulent transactions, but also shows its limitations (low precision, low recall) due to class imbalance.

---

## 📊 Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 rows, 31 columns
- **Features:** 28 PCA‑transformed features (`V1` … `V28`), `Time` (seconds from first transaction), `Amount` (transaction amount), and `Class` (0 = normal, 1 = fraud)
- **Imbalance:** Only 0.1727% fraud

---

## 🧠 Methodology

1. **Exploratory Data Analysis (EDA)**  
   - Checked for missing values (none)  
   - Analyzed distribution of `Amount` and `Time` for normal vs fraud transactions  
   - Visualised with histograms and boxplots (log scale for amount)

2. **Anomaly Detection Rule – Z‑score**  
   - Calculated mean and standard deviation of `Amount`  
   - Flagged transaction as anomaly if:  
     \[
     |\text{Amount} - \mu| > 3 \times \sigma
     \]  
   - Added a new column `amount_anomaly` (1 = anomalous, 0 = normal)

3. **Evaluation**  
   - Compared anomaly flags against true `Class` labels  
   - Computed confusion matrix, accuracy, precision, recall, and F1‑score

4. **Optional improvement**  
   - Used a percentile‑based threshold (top 1% of amounts) to increase recall

---

## 📈 Results (Z‑score method)

| Metric     | Value (example) |
|------------|----------------|
| Accuracy   | ~98.9%         |
| Precision  | ~2.3%          |
| Recall     | ~30.5%         |
| F1‑score   | ~0.043         |

> *Note: Your exact numbers may vary slightly due to random or environment factors.*

- **High accuracy** is misleading because the dataset is extremely imbalanced (99.83% normal).  
- **Low precision** means many flagged anomalies are actually normal large purchases (e.g., a car).  
- **Low recall** means most frauds have small amounts and are missed.

These limitations highlight why **more advanced methods** (e.g., Isolation Forest, SMOTE + Logistic Regression) are needed for this task.

---

## 🚀 How to Run the Project

### Option 1: Google Colab (recommended)

1. Click the badge below to open the notebook directly in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoqueariful/credit-card-fraud-detection/blob/main/Credit_Card_Fraud_Detection.ipynb)

   *(If the badge doesn’t work, manually upload the notebook to Colab.)*

2. Run all cells in order. The first cells will download the dataset automatically using the Kaggle API.

### Option 2: Local Jupyter Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/hoqueariful/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
