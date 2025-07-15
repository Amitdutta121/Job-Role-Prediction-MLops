import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split

from logger_setup import setup_logger
import logging

from src import get_project_root, constants
from src.utils import save_train_test_data, saving_data

# get file name
logger = setup_logger(__name__, log_file='data_loader.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from  a file"""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise




if __name__ == '__main__':
    test_size = 0.2
    data_path = os.path.join(get_project_root(), constants.Folders.data)
    final_df = load_data('https://raw.githubusercontent.com/Amitdutta121/Job-Role-Prediction-MLops/3999f1933732fb8e07e9fea4bfd9f79a0f7b9d97/data/UpdatedResumeDataSet.csv')
    logger.debug('Data loaded successfully from the URL.')
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)

    saving_data(train_data, test_data, 'loaded')
    logger.debug('Train and test data saved successfully.')
