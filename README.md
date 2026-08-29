# heart-disease-risk-prediction
ML pipeline predicting heart disease risk from symptoms &amp; lifestyle data — 11 models + stacking ensemble, ROC/PR analysis, 5-fold CV, and feature importance (Python, Scikit-learn, XGBoost, LightGBM, CatBoost)
# ❤️ Heart Disease Risk Prediction

Predicting early heart-disease risk from self-reported symptoms and lifestyle factors, using classical ML, boosting models, and a stacking ensemble — with full model comparison, ROC/PR analysis, cross-validation, and feature importance.

## 📌 Overview

This project builds and compares **11 machine learning models** (plus a stacking ensemble) to predict whether a person is at risk of heart disease, based on 17 binary symptom/lifestyle features and age. The goal is to explore which symptoms are most predictive and how far a purely classical ML pipeline can go on a well-structured tabular health dataset.

## 📊 Dataset

- **Source:** [Heart Disease Risk Prediction Dataset](https://www.kaggle.com/datasets/ashfakur1389/heart-disease-risk-prediction-2) (Kaggle)
- **Size:** 70,000 rows × 19 columns
- **Target:** `Heart_Risk` (0 = not at risk, 1 = at risk) — balanced 50/50
- **Features:** 15 binary symptom/lifestyle flags (Chest_Pain, Shortness_of_Breath, Fatigue, Palpitations, Dizziness, Swelling, Pain_Arms_Jaw_Back, Cold_Sweats_Nausea, High_BP, High_Cholesterol, Diabetes, Smoking, Obesity, Sedentary_Lifestyle, Family_History, Chronic_Stress), plus `Gender` and `Age`
- Contains 6,245 duplicate rows and no missing values

> ⚠️ Note: this dataset's features determine the label in an almost rule-based way, so near-100% scores are expected here and shouldn't be read as "production-ready for real diagnosis." On messier, real-world clinical data, expect meaningfully lower and more spread-out scores.

## ⚙️ Approach

1. **EDA** — correlation heatmap, class distribution, duplicate/missing-value check
2. **Preprocessing** — dropped `Gender` (not predictive), stratified 70/15/15 train/validation/test split
3. **Modeling** — trained and tuned 11 models:
   Logistic Regression, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, SVC
4. **Stacking Ensemble** — combined Random Forest, Extra Trees, XGBoost, LightGBM, and CatBoost as base learners with a Logistic Regression meta-learner (5-fold stacking CV)
5. **Evaluation** — accuracy, precision, recall, F1, confusion matrices, ROC-AUC, PR-AUC for every model
6. **Robustness check** — 5-fold stratified cross-validation across all base models
7. **Explainability** — feature importance from the best tree-based model
8. **Deployment prep** — best model persisted with `joblib` for later inference

## 🏆 Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Stacking Ensemble** ⭐ | 0.9928 | 0.9943 | 0.9912 | 0.9928 | 0.9996 |
| Logistic Regression | 0.9910 | 0.9896 | 0.9924 | 0.9910 | — |
| SVC | 0.9899 | 0.9873 | 0.9926 | 0.9899 | — |
| Random Forest | 0.9848 | 0.9849 | 0.9846 | 0.9848 | — |
| Gradient Boosting | 0.9826 | 0.9728 | 0.9930 | 0.9828 | — |

Full metrics (all 12 models, ROC/PR curves, and 5-fold CV stability) are in the notebook.

## 🗂️ Project Structure

```
├── heart_disease_risk_prediction.ipynb   # Main notebook (EDA → modeling → evaluation)
├── heart_disease_risk_dataset_earlymed.csv
├── best_heart_risk_model.pkl             # Saved best model (generated after running the notebook)
└── README.md
```

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## ▶️ How to Run

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook heart_disease_risk_prediction.ipynb
```

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
joblib
```

## 👤 Author

**Ashfakur Rahman**
[LinkedIn](https://www.linkedin.com/in/ashfakur-rahman-ramin-994222432/) · [GitHub](https://github.com/ashfakurrahman221-ops) · [Kaggle](https://www.kaggle.com/rahmanashfakur)

## 📄 License

This project is open-source under the [MIT License](LICENSE).
