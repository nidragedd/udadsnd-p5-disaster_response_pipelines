"""
Python script to call to launch preprocessing steps:
    1. Loads the messages and categories datasets
    2. Merges the two datasets
    3. Data cleaning
    4. Save cleaned data in a SQLite database

@author: nidragedd
"""
import sys
import os
import logging
import pandas as pd
from sqlalchemy import create_engine

# This hack is mandatory in order to be able to import modules from sibling packages
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from src.config import utils
from src.config.pgconf import ProgramConfiguration


def load_data():
    """
    Load and merge both datasets (paths and names are loaded from external JSON configuration file)
    :return: (pandas DataFrame) a single dataframe which is the result of the merge operation between both datasets
    """
    messages = pd.read_csv(utils.pgconf.get_messages_raw_file())
    categories = pd.read_csv(utils.pgconf.get_categories_raw_file())
    return categories.merge(messages, on='id', how='inner')


def clean_data(df):
    """
    * Perform some string manipulations on categories dataset:
       1. Create a new DataFrame with individual category columns
       2. Use first row of the categories dataframe to extract a list of new column names for categories.
       3. Rename the columns of `categories`
       4. Convert category values to just numbers 0 or 1.
    * Drop duplicates
    * Remove useless features
    :param df: (pandas DataFrame) the dataset to clean
    :return: (pandas DataFrame) a new cleaned dataset
    """
    logger.info("Cleaning is starting, there are {} rows and {} columns in the dataset".format(df.shape[0], df.shape[1]))
    # Steps 1 to 3
    categories_split_df = df.categories.str.split(';', expand=True)
    category_colnames = categories_split_df.iloc[0].str[:-2]
    categories_split_df.columns = category_colnames

    # Check it worked as expected
    assert categories_split_df.shape[1] == len(category_colnames)

    # To numeric conversion
    categories_split_df = categories_split_df.apply(lambda x: x.str[-1:].astype(int))

    # Drop 'categories' and 'original' columns from merged dataset
    new_df = df.drop(['categories', 'original'], axis=1)
    # Then concatenate new `categories` dataframe
    new_df = pd.concat([new_df, categories_split_df], axis=1)

    # Check the operation worked as expected
    assert new_df.shape[1] == pd.read_csv(utils.pgconf.get_messages_raw_file()).shape[1] - 1 + categories_split_df.shape[1]

    # Get the number of duplicates that should be removed after the operation and number of rows before the operation
    # Drop duplicates and then check it worked as expected
    nb_dups = new_df[new_df.duplicated()].shape[0]
    nb_rows_before_drop_dup = new_df.shape[0]
    new_df.drop_duplicates(inplace=True)

    assert nb_rows_before_drop_dup - new_df.shape[0] == nb_dups

    logger.info("After duplicates removal, there are {} rows and {} columns in the dataset".
                format(new_df.shape[0], new_df.shape[1]))

    # Feature 'child_alone' removal as there are no sample for this category
    new_df = new_df.drop(['child_alone'], axis=1)
    # Convert rows where related=2 to related=0
    new_df.loc[(new_df.related == 2), 'related'] = 0

    logger.info("Cleaning is over, there are {} rows and {} columns in the dataset".
                format(new_df.shape[0], new_df.shape[1]))

    return new_df


def save_data(df):
    """
    Save the cleaned dataframe to a local SQLite DB - A check is made to ensure that save operation worked as expected
    :param df: (pandas DataFrame) the data to save
    """
    logger.info("Saving clean dataframe to '{}' SQLite DB".format(utils.pgconf.get_database_file()))
    engine = create_engine('sqlite:///{}'.format(utils.pgconf.get_database_file()))

    df.to_sql(utils.pgconf.get_db_table_name(), engine, index=False, if_exists='replace')
    df_read = pd.read_sql_table(utils.pgconf.get_db_table_name(), engine)
    logger.info("Control check: {} rows read in DB vs. {} rows to save in dataframe".
                format(df_read.shape[0], df.shape[0]))


logger = logging.getLogger()

if __name__ == '__main__':
    # Handle mandatory arguments
    args = utils.parse_args()
    config_file = vars(args)['config_file']
    log_config_file = vars(args)['log_file']

    # Configure the whole program (logging, external config files, singletons, ...)
    utils.configure_logging(log_config_file)
    utils.pgconf = ProgramConfiguration(config_file)

    save_data(clean_data(load_data()))
