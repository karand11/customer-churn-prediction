

# 🚀 **Customer Churn Prediction — Production-Style ML Pipeline**

End-to-end machine learning system that predicts telecom customer churn and translates model outputs into actionable business retention strategies.

Built with a strong focus on **reproducibility, modular pipelines, model evaluation, and real-world deployment readiness**.

---

## 🧠 **Why This Project Matters**

Customer churn directly impacts revenue in subscription businesses.
This system enables companies to:

* Identify high-risk customers early
* Understand drivers behind churn
* Launch targeted retention campaigns

> Retaining existing customers is significantly cheaper than acquiring new ones — predictive retention directly improves profitability.

---

## 🏗️ **What I Built**

### ✔ **End-to-End ML Workflow**

* Data ingestion & preprocessing pipeline
* Feature engineering experiments
* Multi-model training & comparison
* Model evaluation with business interpretation
* Reusable prediction interface
* Production-style project structure

---

### ✔ **Engineering Highlights**

* Modular pipeline architecture (`src/`)
* Serialized preprocessing + model artifacts
* Reproducible training workflow
* Separation of notebooks vs production code
* Model comparison with standardized metrics
* Business-driven evaluation beyond accuracy

---

## 📊 **Dataset**

**IBM Telco Customer Churn Dataset**
7,043 telecom customers with 20+ behavioral & financial features.

### **Feature Categories**

* Demographics
* Service subscriptions
* Contract & billing information
* Usage patterns
* Monthly & total charges

**Target:** Customer Churn (binary classification)

Class imbalance present (~26% churn).

---

## ⚙️ **ML Approach**

### **Data Pipeline**

* Missing value handling
* Categorical encoding
* Feature scaling
* Train/test splitting
* Persisted preprocessing pipeline

---

### **Models Evaluated**

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

Evaluation Metrics:

* Precision / Recall
* F1 Score
* ROC-AUC
* Confusion Matrix Analysis

---

## 📈 **Results**

| Model               | Accuracy | Precision | Recall   | F1       | ROC-AUC  |
| ------------------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.80     | 0.67      | 0.55     | 0.60     | 0.85     |
| Decision Tree       | 0.78     | 0.61      | 0.57     | 0.59     | 0.76     |
| **Random Forest**   | **0.82** | **0.71**  | **0.58** | **0.64** | **0.86** |
| Gradient Boosting   | 0.81     | 0.69      | 0.57     | 0.63     | 0.85     |
| XGBoost             | 0.82     | 0.70      | 0.58     | 0.64     | 0.86     |

### **Best Model: Random Forest**

* ROC-AUC: 0.8598
* Precision: 71%
* Recall on churners: 58%

Trade-off: Increasing recall reduces missed churners but increases false positives — an important business decision.

---

## 💡 **Key Business Insights**

* Month-to-month contracts show highest churn risk
* First 12 months are critical retention window
* Fiber customers demonstrate elevated churn
* Higher monthly charges correlate with churn
* Tech support significantly reduces churn probability

---

## 🚀 **How to Run**

### **Setup**

```bash
git clone https://github.com/karand11/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv venv
venv\Scripts\activate
# or
source venv/bin/activate

pip install -r requirements.txt
```

Place dataset in:

```
data/raw/
```

---

### **Training Pipeline**

```bash
python src/data_preprocessing.py
python src/train.py
python src/evaluate.py
```

---

### **Generate Predictions**

```bash
python src/predict.py
```

Programmatic usage:

```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor()
prediction, probability = predictor.predict(customer_data)
```

---

## 📁 **Project Structure**

```
customer-churn-prediction/
├── data/
├── notebooks/
├── src/
├── models/
├── results/
├── reports/
└── README.md
```

---

## 🛠️ **Tech Stack**

* Python
* pandas / NumPy
* scikit-learn
* XGBoost
* matplotlib / seaborn
* Jupyter

---

## 🔮 **Future Work**

* Hyperparameter optimization
* SMOTE & imbalance handling
* Model explainability (SHAP)
* REST API deployment
* Docker containerization
* Streamlit dashboard
* Cloud deployment (AWS/GCP)

---

## 👤 **Author**

**Karan Dhanawade**
Master’s Student — Computer Science (Data Engineering)
TU Chemnitz

GitHub: https://github.com/karand11/customer-churn-prediction.git
LinkedIn: https://linkedin.com/in/karan-dhanawade

---

