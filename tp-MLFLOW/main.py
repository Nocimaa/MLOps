import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["AWS_ENDPOINT_URL"] = "http://mlopstp2-minio-7zy4sj-c8285d-51-91-77-165.traefik.me"

# Load data
df = pd.read_csv("insurance.csv")
print(df.head())

X = df.drop("charges", axis=1)
y = df["charges"]

categorical_features = ["sex", "smoker", "region"]
numeric_features = ["age", "bmi", "children"]

categorical_transformer = Pipeline(
    steps=[("encoder", OneHotEncoder(drop="first", sparse_output=False))]
)
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

pipeline_lr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
pipeline_lr.fit(X_train, y_train)

# Evaluate
r2_score = pipeline_lr.score(X_test, y_test)
print(f"Model R²: {r2_score:.4f}")

# MLflow tracking
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("MlEmb-Insurance")

with mlflow.start_run():
    mlflow.log_metric("r2_score", r2_score)

    # Save model file and log it
    joblib.dump(pipeline_lr, "model.pkl")
    #mlflow.log_artifact("model.pkl", artifact_path="model")

    signature = infer_signature(X_train, pipeline_lr.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=pipeline_lr,
        artifact_path="insurance-model",
        signature=signature,
        input_example=X_train,
        registered_model_name="InsuranceModel"
    )

    mlflow.set_tag("Training Info", "Basic RandomForest model for insurance data")
