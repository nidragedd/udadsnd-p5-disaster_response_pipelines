"""
Helper file used to load and configure external configuration files

@author: nidragedd
"""
import os
import json
import logging.config
import argparse

# Single ref to configuration over the whole program
pgconf = None


def configure_logging(log_config_file):
    """
    Setup logging configuration (il file cannot be loaded or read, a fallback basic configuration is used instead)
    :param log_config_file: (string) path to external logging configuration file
    """
    if os.path.exists(log_config_file):
        with open(log_config_file, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)


def parse_args():
    """
    Parse main program arguments to ensure everything is correctly launched
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True,
                        help="External configuration file (JSON format)")
    parser.add_argument('-l', '--log-file', required=True,
                        help="External logging configuration file (JSON format)")
    return parser.parse_args()
