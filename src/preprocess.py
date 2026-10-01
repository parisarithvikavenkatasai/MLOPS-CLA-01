import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def clean_raw_data(df, id_cols=None):

    df = df.copy()

    if id_cols:
        for col in id_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

    # Handle missing delay values
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

    return df


def engineer_features(df):

    df = df.copy()

    # Total delay
    df["TotalDelay"] = (
        df["Departure Delay in Minutes"]
        + df["Arrival Delay in Minutes"]
    )

    # 14 service features
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

    existing_columns = [
        col for col in service_columns
        if col in df.columns
    ]

    # Average service score
    df["CompositeServiceScore"] = (
        df[existing_columns].mean(axis=1)
    )

    return df


def encode_target(target):

    return target.map({
        "neutral or dissatisfied": 0,
        "satisfied": 1
    })


def build_preprocessor(cat_cols, num_cols):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                num_cols
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                cat_cols
            )
        ]
    )

    return preprocessor