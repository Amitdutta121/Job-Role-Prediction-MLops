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


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> Tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
    """
    Apply TF-IDF vectorization to the 'cleaned_resume' column of train and test datasets,
    and return DataFrames with TF-IDF features and the original 'Category' column appended.

    Returns:
        train_df (pd.DataFrame): TF-IDF features with target for training data.
        test_df (pd.DataFrame): TF-IDF features with target for test data.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

        # Vectorize text
        train_tfidf = vectorizer.fit_transform(train_data['cleaned_resume'])
        test_tfidf = vectorizer.transform(test_data['cleaned_resume'])

        # Convert to DataFrames
        feature_names = vectorizer.get_feature_names_out()
        train_df = pd.DataFrame(train_tfidf.toarray(), columns=feature_names)
        test_df = pd.DataFrame(test_tfidf.toarray(), columns=feature_names)

        # Append 'Category' column if it exists
        if 'Category' in train_data.columns:
            train_df['Category'] = train_data['Category'].values
        if 'Category' in test_data.columns:
            test_df['Category'] = test_data['Category'].values

        logger.debug("TF-IDF applied successfully with %d features", max_features)
        return train_df, test_df, vectorizer

    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


if __name__ == '__main__':
    # Load the data
    logger.debug('Starting feature engineering process.')

    processed_train_df, processed_test_df = loading_data('processed')

    logger.debug('Data loaded successfully.')
    max_features = 7000  # You can change this value based on your requirements

    train_df, test_df, _ = apply_tfidf(processed_train_df, processed_test_df, max_features)
    logger.debug('TfIdf applied successfully.')
    # Save the transformed data
    saving_data(train_df, test_df, 'interim')

    logger.debug('Transformed data saved successfully.')
