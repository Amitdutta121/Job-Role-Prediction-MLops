import pandas as pd
from logger_setup import setup_logger
import logging
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
    df = load_data('https://raw.githubusercontent.com/Amitdutta121/Job-Role-Prediction-MLops/refs/heads/main/data/UpdatedResumeDataSet.csv')
    logger.info('Data loaded successfully with shape: %s', df.shape)