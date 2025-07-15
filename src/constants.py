import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DATA_SOURCE = "https://raw.githubusercontent.com/Amitdutta121/Job-Role-Prediction-MLops/refs/heads/main/data/UpdatedResumeDataSet.csv"

# constants.py
from types import SimpleNamespace

LOG_FOLDER = "logs"
REPORT_FOLDER = "reports"
DATA_FOLDER = "data"
MODEL_FOLDER = "models"
CONFIG_FILE = "config.json"
PARAMS_FILE = "params.yaml"
RAW = 'raw'
INTERIM = 'interim'
PROCESSED = 'processed'


class LogFiles:
    data_ingestion = "data_ingestion.log"
    model_building = "model_building.log"
    model_evaluation = "model_evaluation.log"
    data_preprocessing = "data_preprocessing.log"
    feature_engineering = "feature_engineering.log"


class Folders:
    logs = LOG_FOLDER
    reports = REPORT_FOLDER
    data = DATA_FOLDER
    models = MODEL_FOLDER
    config = CONFIG_FILE
    params = PARAMS_FILE      # ← FIXED: removed comma
    raw = RAW                 # ← FIXED
    interim = INTERIM         # ← FIXED
    processed = PROCESSED     # ← FIXED

# MLflow Constants
class MLflow:
    experiment_name = "JobRolePredictionExperiment--RandomForest"
    run_training = "Training_RandomForest"
    run_evaluation = "Evaluation_RandomForest"
    tracking_uri = "http://127.0.0.1:5000"  # ✅ Valid local folder
