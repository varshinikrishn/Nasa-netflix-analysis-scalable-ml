# Scalable Machine Learning – Project Report & Codebase

This repository contains all code and analysis completed for the **COM6012 – Scalable Machine Learning** module. The project covers log analysis, regression, classification, recommender systems, clustering, and ensemble learning using large-scale datasets and distributed computing techniques.

---

## 📁 Project Structure

* `Q1/` – Web server log analysis (Germany, Canada, Singapore)
* `Q2/` – Poisson & Logistic Regression on insurance dataset
* `Q3/` – Classification using Random Forest, Gradient Boosting, and Neural Networks on the HIGGS dataset
* `Q4/` – ALS Recommender System + K-means clustering on MovieLens

A detailed explanation of all findings is available in the included PDF report:
**AS_report.pdf**

---

## 🧠 Summary of Tasks

### **Q1 – Log File Analysis**

Performed country-wise traffic analysis using distributed data processing techniques.

Key Outputs:

* Total request counts per country
* Unique hosts per country
* Top 9 frequent hosts
* Daily activity heatmaps
* Inferred patterns in traffic concentration and temporal distribution
* Interpretations relevant to NASA server optimization

---

### **Q2 – Regression & Classification**

Models implemented:

* Poisson Regression
* Logistic Regression (L1 & L2 Regularization)

Metrics & Insights:

* Poisson RMSE ≈ **0.355**
* Logistic Regression AUC ≈ **0.627** (both L1 & L2)
* Accuracy ≈ **88.95%** (but affected by class imbalance)
* Discussion of sparsity, regularization effects, and model interpretability

---

### **Q3 – Ensemble Methods & Neural Networks (HIGGS Dataset)**

Models tested:

* **Random Forest**

  * Accuracy: 0.702
  * AUC: 0.777
* **Gradient Boosting**

  * Accuracy: 0.717 *(Best)*
  * AUC: 0.793 *(Best)*
* **Neural Network**

  * Accuracy: 0.681
  * AUC: 0.738

Gradient Boosting performed best due to its ability to reduce bias and variance with sequential learning.

---

### **Q4 – ALS Recommender System + Clustering**

#### **Task A: ALS**

Two hyperparameter settings were compared.
Improved performance observed with:

* **rank = 20**
* **maxIter = 15**
* **regParam = 0.02**

RMSE/MSE/MAE reported for 40%, 60%, and 80% train-test splits.

#### **Task B: Clustering**

* K-means clustering on user–item interaction data
* Cluster sizes increase proportionally with training data
* Extracted top genres from highly rated movies across all splits
* Key finding: Drama & Comedy consistently dominate user preferences

---

## 🛠️ Technologies & Tools

* **PySpark** for distributed computation
* **Python (NumPy, pandas, scikit-learn)** for ML modelling
* **Matplotlib / Seaborn** for visualization
* **MovieLens** & **HIGGS** datasets
* **Log file parsing** for large-scale text processing

---

## 📘 How to Run

1. clone the repository

   ```
   git clone <your-repo-link>
   ```
2. open project in your environment (Jupyter/VSCode)
3. run notebooks or python files for each question
4. ensure PySpark & dataset paths are configured correctly

---

## 📄 Report

The complete written analysis is contained in **AS_report.pdf**, documenting:

* Data processing steps
* Model training details
* Comparative evaluation
* Interpretations & insights

---

## ✨ Author

**Varshini K**
COM6012 – Scalable Machine Learning
University of Sheffield

---

If you have suggestions for improvements or want to reuse parts of this project, feel free to open an issue or fork the repository.
