import mlflow
from mlflow.tracking import MlflowClient


# ==========================================
# 1. MLFLOW SETUP
# ==========================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

experiment_name = "Airline_Passenger_Satisfaction"

model_name = "Airline_Passenger_Satisfaction_Model"


# ==========================================
# 2. GET EXPERIMENT
# ==========================================

client = MlflowClient()

experiment = client.get_experiment_by_name(
    experiment_name
)

if experiment is None:

    print("Experiment not found.")

    exit()


print()
print("Experiment found:")
print(experiment.name)


# ==========================================
# 3. FIND SUCCESSFUL XGBOOST RUN
# ==========================================

runs = client.search_runs(

    experiment_ids=[experiment.experiment_id],

    filter_string=(
        "attributes.status = 'FINISHED' "
        "and params.model = 'XGBoost'"
    ),

    order_by=[
        "attributes.start_time DESC"
    ],

    max_results=1
)


if len(runs) == 0:

    print()
    print("No successful XGBoost run found.")

    exit()


run = runs[0]

run_id = run.info.run_id


print()
print("Successful XGBoost run found:")
print("Run ID:", run_id)


# ==========================================
# 4. MODEL URI
# ==========================================

model_uri = (
    "runs:/"
    + run_id
    + "/model"
)


print()
print("Model URI:")
print(model_uri)


# ==========================================
# 5. REGISTER MODEL
# ==========================================

result = mlflow.register_model(

    model_uri=model_uri,

    name=model_name

)


# ==========================================
# 6. DISPLAY RESULT
# ==========================================

print()
print("==========================================")
print("MODEL REGISTERED SUCCESSFULLY!")
print("==========================================")

print()
print("Model name:")
print(result.name)

print()
print("Model version:")
print(result.version)

print()
print("Run ID:")
print(run_id)