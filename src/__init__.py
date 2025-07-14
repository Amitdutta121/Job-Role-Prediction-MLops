
from utils import get_project_root, from_root
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from logger_setup import setup_logger
import logging
# get file name
logger = setup_logger(__name__, log_file='__init__.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')