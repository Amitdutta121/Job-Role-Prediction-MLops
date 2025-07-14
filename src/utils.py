import logging
import os

import pandas as pd
import yaml
import joblib
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Literal

from src import constants
from logger_setup import setup_logger

# get file name
logger = setup_logger(__name__, log_file='data_preprocessing.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')


# -------------------------------
# Config Handling
# -------------------------------
def load_yaml(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def from_root(*parts):
    """Build an absolute path from project root."""
    return os.path.join(get_project_root(), *parts)


# -------------------------------
# Timer Decorator
# -------------------------------
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[⏱️] {func.__name__} executed in {time.time() - start:.2f}s")
        return result

    return wrapper


# -------------------------------
# Save / Load Model
# -------------------------------
def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def save_train_test_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, name_train="train.csv",
                         name_test="test.csv") -> None:
    """Save train and test datasets directly to the specified data_path (no 'raw' appended)."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, name_train), index=False)
        test_data.to_csv(os.path.join(data_path, name_test), index=False)
        logger.debug('Train and test data saved to %s', data_path)
    except Exception as e:
        logger.error('Error saving processed data: %s', e)
        raise


def save_data(data: pd.DataFrame, data_path: str, filename: str) -> None:
    """Save a single dataset to a specified path and filename."""
    try:
        raw_data_path = os.path.join(data_path, constants.Folders.raw)
        os.makedirs(raw_data_path, exist_ok=True)
        file_path = os.path.join(raw_data_path, filename)
        data.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def saving_data(final_train_df: pd.DataFrame, final_test_df: pd.DataFrame, saving_type: str) -> None:
    try:
        if saving_type == 'loaded':
            data_path = os.path.join(get_project_root(), constants.Folders.data, constants.Folders.raw)
            save_train_test_data(final_train_df, final_test_df, data_path, name_train="train.csv", name_test="test.csv")
        elif saving_type == 'processed':
            data_path = os.path.join(get_project_root(), constants.Folders.data, constants.Folders.processed)
            save_train_test_data(final_train_df, final_test_df, data_path, name_train="train_processed.csv", name_test="test_processed.csv")
        elif saving_type == 'interim':
            data_path = os.path.join(get_project_root(), constants.Folders.data, constants.Folders.interim)
            save_train_test_data(final_train_df, final_test_df, data_path, name_train="train_interim.csv", name_test="test_interim.csv")
        else:
            raise ValueError(f"Unknown saving type: {saving_type}. Expected 'loaded', 'processed', or 'interim'.")
    except Exception as e:
        logger.error('Failed to save data: %s', e)
        raise


def loading_data(data_type: Literal['raw', 'processed', 'interim']) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if data_type == 'raw':
            data_path = os.path.join(get_project_root(), constants.Folders.data, constants.Folders.raw)
            train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
            test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
        elif data_type == 'processed':
            data_path = os.path.join(get_project_root(), constants.Folders.data, constants.Folders.processed)
            train_data = pd.read_csv(os.path.join(data_path, "train_processed.csv"))
            test_data = pd.read_csv(os.path.join(data_path, "test_processed.csv"))
        elif data_type == 'interim':
            data_path = os.path.join(get_project_root(), constants.Folders.data, constants.Folders.interim)
            train_data = pd.read_csv(os.path.join(data_path, "train_processed.csv"))
            test_data = pd.read_csv(os.path.join(data_path, "test_processed.csv"))
        else:
            raise ValueError(f"Unknown data type: {data_type}. Expected 'raw', 'processed', or 'interim'.")
        logger.debug('Data loaded successfully from %s', data_path)
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise


# -------------------------------
# Save / Load JSON
# -------------------------------
def save_json(data: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


# -------------------------------
# Path Helper
# -------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]
