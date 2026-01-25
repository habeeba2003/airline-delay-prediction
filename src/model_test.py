import joblib
import pandas as pd

# -----------------------------
# Load models & scalers
# -----------------------------
clf = joblib.load("xgb_classifier_model.joblib")
scaler_class = joblib.load("scaler_classifier.joblib")

reg = joblib.load("xgb_regressor_model.joblib")
scaler_reg = joblib.load("scaler_regressor.joblib")

print("Models Loaded Successfully!\n")

print("Enter flight details below (don't enter DEP_DELAY or DELAYED):\n")

# -----------------------------
# USER INPUT
# -----------------------------
fields = {
    "DAY_OF_MONTH": int,
    "DAY_OF_WEEK": int,
    "OP_UNIQUE_CARRIER": int,
    "DEST": int,
    "CRS_ELAPSED_TIME": float,
    "DISTANCE": float,
    "CRS_DEP_M": float,
    "DEP_TIME_M": float,
    "CRS_ARR_M": float,
    "Temperature": float,
    "Dew Point": float,
    "Humidity": float,
    "Wind": float,
    "Wind Speed": float,
    "Pressure": float,
    "Condition": int,
    "sch_dep": int,
    "sch_arr": int,
    "TAXI_OUT": float,
    "DEP_HOUR": int,
    "IS_WEEKEND": int,
    "ROUTE": int,
    "TIME_SLOT": int
}

user_data = {}
for key, tp in fields.items():
    user_data[key] = tp(input(f"{key}: "))

new_data = pd.DataFrame([user_data])

# -----------------------------
# CLASSIFICATION
# -----------------------------
scaled_class = scaler_class.transform(new_data)
pred_class = clf.predict(scaled_class)[0]
pred_prob = clf.predict_proba(scaled_class)[0][1]

classifier_status = "DELAYED" if pred_class == 1 else "ON-TIME"

# -----------------------------
# REGRESSION
# -----------------------------
scaled_reg = scaler_reg.transform(new_data)
delay_minutes = reg.predict(scaled_reg)[0]



# -----------------------------
# FINAL PROJECT DECISION
# -----------------------------
print("\n-----------------------------------")
print("🚦 FINAL PROJECT DECISION")
print("-----------------------------------")

if delay_minutes <= 5:
    final_status = "ON-TIME"
else:
    final_status = "DELAYED"

print(f"Flight Status: {final_status}")
print(f"Predicted Delay: {delay_minutes:.2f} minutes")
print("-----------------------------------")
