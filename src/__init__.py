import logging

from logger_setup import setup_logger
from utils import get_project_root, from_root

# get file name
logger = setup_logger(__name__, log_file='__init__.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')
