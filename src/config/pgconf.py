"""
Class used to handle program configuration

@author: nidragedd
"""
import logging
import os
import json


class ProgramConfiguration(object):
    """
    Class used to handle and maintain all parameters of this program (timeouts, some other values...)
    """
    _config = None
    _config_directory = None
    _logger = logging.getLogger()

    def __init__(self, config_file_path):
        """
        Constructor - Loads the given external JSON configuration file. Raises an error if not able to do it.
        :param config_file_path: (string) full path to the JSON configuration file
        """
        if os.path.exists(config_file_path):
            self._config_directory = os.path.dirname(config_file_path)
            with open(config_file_path, 'rt') as f:
                self._config = json.load(f)
                self._logger.info("External JSON configuration file ('{}') successfully loaded".format(config_file_path))
        else:
            raise Exception("Could not load external JSON configuration file '{}'".format(config_file_path))

    def _get_path_to_folder(self, folder_key):
        """
        Get full path to a folder given its key name in the external JSON configuration file
        :param folder_key: key name of the folder as written in JSON configuration file
        :return: (string) path to file to load
        """
        # If path to raw folder is relative to JSON configuration file
        if folder_key.startswith('.'):
            folder_key = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, folder_key)
        if os.path.exists(folder_key):
            return folder_key
        else:
            raise Exception("Directory '{}' does not exist".format(folder_key))

    def _get_raw_file(self, filename):
        """
        Get full path to a RAW file given its filename. Path is retrieved from external JSON configuration file
        :param filename: key name of the file as written in JSON configuration file
        :return: (string) path to file to load
        """
        raw_folder = self._get_path_to_folder(self._config['data']['raw_input_folder'])
        return os.path.join(raw_folder, self._config['data'][filename])

    def get_messages_raw_file(self):
        """
        Retrieve the path to RAW messages CSV file
        :return: (string) path to RAW messages CSV file
        """
        return self._get_raw_file('messages_file')

    def get_categories_raw_file(self):
        """
        Retrieve the path to RAW categories CSV file
        :return: (string) path to RAW categories CSV file
        """
        return self._get_raw_file('categories_file')

    def get_database_file(self):
        """
        :return: (string) database file name as configured in the external JSON configuration file
        """
        output_folder = self._get_path_to_folder(self._config['database']['output_folder'])
        return os.path.join(output_folder, self._config['database']['db_name'])

    def get_db_table_name(self):
        """
        :return: (string) database main table name as configured in the external JSON configuration file
        """
        return self._config['database']['messages_table_name']
