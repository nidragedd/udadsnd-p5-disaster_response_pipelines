"""
Python file used to load and hold data across the program

@author: nidragedd
"""
import logging
import pandas as pd
from sqlalchemy import create_engine

from src.config import utils

logger = logging.getLogger()
df = None


def load_data():
    """
    Load the data from the SQLite DB file configured in external JSON configuration file
    :return:
    """
    logger.info("Loading data from '{}' SQLite DB".format(utils.pgconf.get_database_file()))
    engine = create_engine('sqlite:///{}'.format(utils.pgconf.get_database_file()))
    try:
        return pd.read_sql_table(utils.pgconf.get_db_table_name(), engine), True
    except ValueError:
        error_msg = "Not able to load data from '{}' SQLite DB.".format(utils.pgconf.get_database_file())
        error_msg += " Have you launched the 'process_data.py' script to build it?"
        error_msg += " If yes then please check your path in external configuration JSON file"
        logger.error(error_msg)
        return None, False
