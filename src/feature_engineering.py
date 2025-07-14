import pandas as pd
from logger_setup import setup_logger
import logging
from typing import Tuple

from src import get_project_root
from src.data_loader import load_data
from src.utils import loading_data, saving_data

# get file name
logger = setup_logger(__name__, log_file='data_loader.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')
from sklearn.feature_extraction.text import TfidfVectorizer
import constants
import os


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply TF-IDF vectorization to the cleaned_resume column of train and test data."""
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)  # unigrams and bigrams
        )

        train_tfidf = vectorizer.fit_transform(train_data['cleaned_resume'])
        test_tfidf = vectorizer.transform(test_data['cleaned_resume'])

        # Convert the sparse matrix to a DataFrame for compatibility
        train_df = pd.DataFrame(train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        test_df = pd.DataFrame(test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

        logger.debug("TF-IDF applied with %d features", max_features)
        return train_df, test_df

    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


if __name__ == '__main__':
    # Load the data
    logger.debug('Starting feature engineering process.')

    processed_train_df, preprocess_test_df = loading_data('processed')
    if processed_train_df is None or preprocess_test_df is None:
        logger.error('Failed to load the data. Exiting feature engineering process.')
        raise ValueError("Data loading failed. Please check the data source.")
    logger.debug('Data loaded successfully.')
    max_features = 7000  # You can change this value based on your requirements

    train_df, test_df = apply_tfidf(processed_train_df, preprocess_test_df, max_features)
    logger.debug('TfIdf applied successfully.')
    # Save the transformed data
    saving_data(train_df, test_df, 'interim')

    logger.debug('Transformed data saved successfully.')
