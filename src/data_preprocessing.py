import pandas as pd
import string
from logger_setup import setup_logger
import logging

# get file name
logger = setup_logger(__name__, log_file='data_preprocessing.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')


def clean_text(text: str) -> str:
    try:
        text = text.lower()  # Convert to lowercase
        text = text.replace('\n', ' ')  # Replace newlines
        text = text.replace('\r', ' ')  # Replace carriage returns
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = ''.join([ch for ch in text if not ch.isdigit()])  # Remove digits
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise e


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """CLeanning the data"""
    try:
        df['cleaned_resume'] = df['Resume'].apply(clean_text)
        return df
    except KeyError as e:
        logger.error(f"Column not found in DataFrame: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error cleaning DataFrame: {e}")
        raise e
