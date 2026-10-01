import pandas as pd
import mlflow


# ============================================================
# 1. MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


# ============================================================
# 2. REGISTERED MODEL
# ============================================================

model_name = "Airline_Passenger_Satisfaction_Model"

model_version = "1"

model_uri = (
    "models:/"
    + model_name
    + "/"
    + model_version
)


print()
print("Loading registered model...")
print("Model:", model_name)
print("Version:", model_version)


# ============================================================
# 3. LOAD MODEL
# ============================================================

model = mlflow.pyfunc.load_model(
    model_uri
)


print()
print("Registered model loaded successfully!")


# ============================================================
# 4. LOAD PASSENGER DATA
# ============================================================

data_path = "data/raw/test.csv.xlsx"

df = pd.read_excel(
    data_path
)


print()
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 5. DATA CLEANING
# ============================================================

columns_to_drop = [
    "Unnamed: 0",
    "id"
]

for col in columns_to_drop:

    if col in df.columns:

        df = df.drop(
            columns=[col]
        )


# ============================================================
# 6. HANDLE DELAY COLUMNS
# ============================================================

delay_columns = [
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes"
]

for col in delay_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = df[col].fillna(0)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

# Total delay

df["TotalDelay"] = (
    df["Departure Delay in Minutes"]
    +
    df["Arrival Delay in Minutes"]
)


# Service columns

service_columns = [

    "Inflight wifi service",

    "Departure/Arrival time convenient",

    "Ease of Online booking",

    "Gate location",

    "Food and drink",

    "Online boarding",

    "Seat comfort",

    "Inflight entertainment",

    "On-board service",

    "Leg room service",

    "Baggage handling",

    "Checkin service",

    "Inflight service",

    "Cleanliness"
]


existing_service_columns = [

    col

    for col in service_columns

    if col in df.columns

]


df["CompositeServiceScore"] = (

    df[existing_service_columns]
    .mean(axis=1)

)


# ============================================================
# 8. REMOVE TARGET
# ============================================================

if "satisfaction" in df.columns:

    X = df.drop(
        columns=["satisfaction"]
    )

else:

    X = df.copy()


# ============================================================
# 9. SELECT ONE PASSENGER
# ============================================================

passenger = X.iloc[[0]].copy()


print()
print("Passenger selected for prediction:")
print(passenger)


# ============================================================
# 10. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    passenger
)


# ============================================================
# 11. DISPLAY RESULT
# ============================================================

print()
print("==========================================")
print("PREDICTION RESULT")
print("==========================================")


if prediction[0] == 1:

    print(
        "Predicted Satisfaction: satisfied"
    )

else:

    print(
        "Predicted Satisfaction: neutral or dissatisfied"
    )


print()
print("Prediction value:", prediction[0])

print()
print("==========================================")
print("Prediction completed successfully!")
print("==========================================")