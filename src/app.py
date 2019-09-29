"""
Flask webapp main file
Steps:
    1. Load external configuration files (logging and program configuration)
    2. Load data from DB
    3. Start the Flask app server

@author: nidragedd
"""
import logging
import sys
import os
# This hack is mandatory in order to be able to import modules from sibling packages
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from src import app
from src.config import utils
from src.config.pgconf import ProgramConfiguration
from src.data import dataloader

logger = logging.getLogger()

if __name__ == '__main__':
    # Handle mandatory arguments
    args = utils.parse_args()
    config_file = vars(args)['config_file']
    log_config_file = vars(args)['log_file']

    # Configure the whole program (logging, external config files, singletons, ...)
    utils.configure_logging(log_config_file)
    utils.pgconf = ProgramConfiguration(config_file)

    # Load and keep data
    dataloader.df, ok = dataloader.load_data()
    dataloader.model, dataloader.classes = dataloader.load_model_and_classes()

    if ok:
        # Launch the Flask app server
        app.run(host='0.0.0.0', port=3001)
