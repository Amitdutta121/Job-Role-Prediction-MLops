import pandas as pd
from logger_setup import setup_logger
import logging
# get file name
logger = setup_logger(__name__, log_file='data_loader.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')
from sklearn.feature_extraction.text import TfidfVectorizer
import constants



def vectorize_data():
