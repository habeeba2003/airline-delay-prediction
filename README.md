# ✈️ Air Travel Timeliness Prediction Using Machine Learning

A machine learning project that predicts flight take-off delays at **JFK International Airport** using both classification and regression models. Built with Random Forest and XGBoost, and deployed via a Streamlit web application.

---

## 📌 Project Overview

Flight delays cause cascading disruptions for passengers, airlines, and airport operations. This project leverages historical flight and weather data to:

- **Classify** whether a flight will be **Delayed** or **On-Time**
- **Predict** the exact **delay duration in minutes**

The system is backed by a user-friendly Streamlit interface for real-time predictions.

---

## 📂 Dataset

- **Source:** [Kaggle – Flight Take-Off Data JFK Airport](https://www.kaggle.com/datasets/deepankurk/flight-take-off-data-jfk-airport)
- **Records:** 28,820 flights
- **Features:** 22 independent variables + 1 target (`DEP_DELAY`)
- **Data includes:** Flight details, airline info, weather conditions (temperature, wind, humidity, pressure), taxi-out time, scheduled departure/arrival times

---

## 🧪 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Class Balancing | SMOTE (imbalanced-learn) |
| Model Saving | Joblib |
| Deployment | Streamlit |
| Dev Environment | Google Colab, VS Code |
| Version Control | Git |

---

## ⚙️ Preprocessing Pipeline

1. **Missing Value Handling** – Numerical features filled with median; categorical with mode
2. **Outlier Removal** – IQR-based removal for `DEP_DELAY` and `TAXI_OUT`
3. **Feature Engineering** – Created `DEP_HOUR`, `IS_WEEKEND`, `ROUTE`, `TIME_SLOT`
4. **Target Variable Creation** – `DELAYED` column derived: 1 if `DEP_DELAY > 5 mins`, else 0
5. **Encoding** – Label Encoding for categorical features (`OP_UNIQUE_CARRIER`, `DEST`, `Wind`, `Condition`, `ROUTE`, `TIME_SLOT`)
6. **Scaling** – StandardScaler applied for model convergence
7. **SMOTE Balancing** – Applied only on training data to address class imbalance (79.6% on-time vs 20.4% delayed)

---

## 🤖 Models

### Random Forest
- Ensemble of decision trees using bootstrapping and majority voting
- Classifier: `n_estimators=300, max_depth=10, min_samples_leaf=5`
- Regressor: `n_estimators=200, max_depth=10`

### XGBoost
- Sequential gradient boosting with regularization
- Classifier: `n_estimators=300, learning_rate=0.03, max_depth=5`
- Regressor: `n_estimators=200, learning_rate=0.05, max_depth=5`

---

## 📊 Results

| Metric | Random Forest | XGBoost |
|---|---|---|
| Classification Accuracy | 79.35% | **88.13%** |
| ROC-AUC Score | 0.800 | **0.886** |
| Regression R² Score | 0.6598 | **0.8353** |
| Training Accuracy | 87.33% | 91.74% |

> ✅ **XGBoost outperformed Random Forest** across all metrics for both classification and regression tasks.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/habeeba2003/flight-delay.git
cd flight-delay
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## 📋 Requirements

```
python>=3.8
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
joblib
matplotlib
seaborn
streamlit
```

## 🖥️ Application UI

The Streamlit app has two pages:

- **Home Page** – Input flight details: airline carrier, destination airport, weather conditions, scheduled times, route, time slot, and more
- **Prediction Page** – Displays:
  - Flight Status: `DELAYED` or `ON-TIME`
  - Estimated Delay Duration (in minutes)

---

## 📈 Key Features Used for Prediction

| Feature | Description |
|---|---|
| `OP_UNIQUE_CARRIER` | Airline carrier code |
| `DEST` | Destination airport |
| `CRS_DEP_M` | Scheduled departure time (mins after midnight) |
| `TAXI_OUT` | Time from gate to runway |
| `Temperature`, `Humidity`, `Wind Speed` | Weather at departure |
| `Condition` | Weather condition (Clear/Rain/Fog/Snow) |
| `ROUTE` | Airline-destination combination |
| `DEP_HOUR`, `IS_WEEKEND`, `TIME_SLOT` | Engineered temporal features |

---

## 🔮 Future Scope

- Integration of real-time weather APIs and live flight data
- Deep learning models (LSTM, Transformer) for temporal pattern capture
- Explainable AI (SHAP values) for model transparency
- Multi-airport expansion beyond JFK
- Cloud deployment for large-scale use
- Mobile app integration for passenger delay alerts

---

## 👩‍💻 Author

**Habeeba M N**  
MCA Student, Department of Computer Applications  
Mar Athanasius College of Engineering, Kothamangalam  
APJ Abdul Kalam Technological University  
2024–2026  
 
---

## 📄 License

This project is for academic purposes. Dataset sourced from Kaggle under its respective license.
