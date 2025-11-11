# SHELL-WEEK-2
# ⚡ EV Range Prediction using Machine Learning

An AI-powered project that predicts the **range (in km)** of Electric Vehicles using real-world EV specifications.  
This model helps estimate how far an EV can travel on a single charge based on its key parameters such as battery capacity, efficiency, power, and drivetrain type.

---

## 🚀 Project Overview

This project uses **regression-based Machine Learning models** (Random Forest, Gradient Boosting, XGBoost, and Linear Regression) to predict EV range.  
The best-performing model achieved **over 96% accuracy (R² Score)** on the test data.

---

## 🧠 Key Features

- Predicts EV range using multiple technical parameters  
- Compares multiple ML algorithms and auto-selects the best performer  
- Includes evaluation metrics: **MAE**, **RMSE**, and **R² Score**  
- Visualizes predicted vs actual range for clarity  
- Fully built in **Google Colab**, optimized for tablet use

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn  
- **Environment:** Google Colab  
- **Model Used:** Random Forest Regressor (best performer)

---

## 📊 Evaluation Metrics

| Metric | Description | Example Result |
|:--------|:-------------|:----------------|
| MAE | Mean Absolute Error | 42.17 |
| RMSE | Root Mean Squared Error | 58.43 |
| R² Score | Coefficient of Determination | 0.9676 |

---

## 🧩 Improvisations Made

- Added **feature selection** to reduce dimensionality and improve accuracy  
- Optimized model with **hyperparameter tuning** for better generalization  
- Implemented **visual comparison** between actual and predicted ranges  
- Ensured **Colab & tablet compatibility** for lightweight usage  
- Used **Pickle-based model saving** for easy future deployment

---

## 📁 Files Included

- `ev_range_prediction.ipynb` → Complete Colab notebook with full workflow  
- `electric_vehicles_spec_2025.csv` → Dataset used for training  
- `ev_best_model.pkl` → Trained ML model  
- `requirements.txt` → Required Python dependencies  
- `README.md` → Project overview and documentation  

---

## 🧠 How to Use

1. Open **Google Colab**  
2. Upload the notebook and dataset  
3. Run all cells sequentially  
4. The best model will be saved as `.pkl`  
5. Use `files.download()` in Colab to download the trained model

---

## 📈 Visualization Example
(You can add this after saving a plot as an image)
```python
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Range (km)")
plt.ylabel("Predicted Range (km)")
plt.title("Actual vs Predicted EV Range")
plt.show()
