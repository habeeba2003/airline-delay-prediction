# ------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns 
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, mean_absolute_error, mean_squared_error,
    r2_score
)
# --- Changed Imports ---
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# -----------------------
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Using your original file load
try:
    # IMPORTANT: Ensure this file path is correct in your environment
    df = pd.read_csv("preprocessed_data.csv")
except FileNotFoundError:
    print("Error: 'preprocessed_data.csv' not found. Using dummy data for demonstration.")
    # --- Dummy DataFrame for demonstration ---
    data_size = 1000
    df = pd.DataFrame({
        'FEATURE_1': np.random.rand(data_size),
        'FEATURE_2': np.random.rand(data_size) * 10,
        'DELAYED': np.random.randint(0, 2, data_size),
        'DEP_DELAY': np.random.uniform(0, 60, data_size)
    })
    # -----------------------------------------


# For classification (0/1 delayed)
target_class = "DELAYED"     

# For regression (delay minutes)
target_reg = "DEP_DELAY"

## ✈️ Classification Model (Random Forest)
# FEATURES & TARGET
X = df.drop([target_class, target_reg], axis=1, errors='ignore')
y = df[target_class]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# APPLY SMOTE (ONLY ON TRAINING DATA) - sampling_strategy='auto' for 1:1 ratio
sm = SMOTE(random_state=42, sampling_strategy='auto') 
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# STANDARDIZATION (Classification)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# RANDOM FOREST CLASSIFIER 
clf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=10,     
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1         
)

clf.fit(X_train_scaled, y_train_res)

# PREDICTIONS
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_test)

print("\n--- Random Forest Classifier Results ---")
print("Training Accuracy:", accuracy_score(y_train_res, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
print("\nConfusion Matrix:\n", cm)

# ----------------------------------------------------
# CONFUSION MATRIX IMAGE PLOT
# ----------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ----------------------------------------------------
# ROC CURVE PLOT
# ----------------------------------------------------
y_prob = clf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="Random Forest ROC (AUC = %.3f)" % roc_auc)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.legend()
plt.show()

# ----------------------------------------------------
# SAVE CLASSIFICATION MODEL & SCALER
# ----------------------------------------------------
joblib.dump(clf, 'rf_classifier_model.joblib')
joblib.dump(scaler, 'scaler_rf_classifier.joblib')
print("--------------------------------------------------")
print("✅ Random Forest Classifier and Scaler saved.")


## ⏱️ Regression Model (Random Forest)

# FEATURES & TARGET
Xr = df.drop([target_reg, target_class], axis=1, errors='ignore')
yr = df[target_reg]

# TRAIN TEST SPLIT
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

# STANDARDIZATION (Regression)
scaler_r = StandardScaler()
Xr_train_scaled = scaler_r.fit_transform(Xr_train)
Xr_test_scaled = scaler_r.transform(Xr_test)

# RANDOM FOREST REGRESSOR 
reg = RandomForestRegressor(
    n_estimators=200,     
    max_depth=10,          
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

reg.fit(Xr_train_scaled, yr_train)

# PREDICTIONS
yr_pred = reg.predict(Xr_test_scaled)

# Calculate standard metrics
r2 = r2_score(yr_test, yr_pred)
mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))

# --- NORMALIZATION OF ERRORS ---
yr_range = yr_test.max() - yr_test.min()

# Calculate Normalized MAE and RMSE
if yr_range > 0:
    nmae = mae / yr_range
    nrmse = rmse / yr_range
else:
    nmae = 0.0
    nrmse = 0.0

print("\n--- Random Forest Regressor Results ---")
print("Regression R2 Score:", r2)
print(f"Normalized MAE (NMAE): {nmae:.4f}")
print(f"Normalized RMSE (NRMSE): {nrmse:.4f}")

# ----------------------------------------------------
# SAVE REGRESSION MODEL & SCALER
# ----------------------------------------------------
joblib.dump(reg, 'rf_regressor_model.joblib')
joblib.dump(scaler_r, 'scaler_rf_regressor.joblib')
print("✅ Random Forest Regressor and Scaler saved.")

print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))
print("Regression R2 Score:", r2)