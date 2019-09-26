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

    def get_output_model_file(self):
        """
        :return: (string) path to output directory where models are dumped (as configured in the external JSON
        configuration file)
        """
        output_folder = self._get_path_to_folder(self._config['modeling']['output_folder'])
        return os.path.join(output_folder, "{}.joblib".format(self.get_chosen_model_name().lower().replace(' ', '-')))

    def get_db_table_name(self):
        """
        :return: (string) database main table name as configured in the external JSON configuration file
        """
        return self._config['database']['messages_table_name']

    def is_model_tuning_active(self):
        """
        :return: True if the model should be trained with GridSearch and config parameters specified in file
        """
        return self._config['modeling']['tune_model']

    def get_chosen_model_name(self):
        """
        :return: (string) name of the model to build. Should be one among available models
        """
        available_models = ["Logistic Regression", "Random Forest"]
        model = self._config['modeling']['model_to_build']
        if model in available_models:
            return model
        else:
            raise Exception("Model '{}' is not valid. Must be one among {}".format(model, available_models))

    def _get_tuning_parameters(self):
        """
        Internal method to retrieve the model tuning parameters within JSON configuration file
        :return: dict of tuning parameters as specified in the file
        """
        return self._config['modeling']['tuning_parameters']

    def handle_tfidf_tuning_param_grid(self, param_grid):
        """
        Add to a given parameter grid the parameters to tune the TF-IDF vectorizer. Values can be a range of values if
        tuning is active (so GridSearch will be performed) or default ones.
        WARNING: this method will mutate the given dict parameter
        :param param_grid: (dict) this method will mutate this object by appending it some keys/values
        :return: key/values dict with parameters set for the TF-IDF vectorizer
        """
        vals = "gridsearch_values" if self.is_model_tuning_active() else "default_values"
        tuning_params = self._get_tuning_parameters()["tf-idf"][vals]

        # Handle easy parameters, just assign values that could be single one value or arrays of numeric or boolean
        for i in ['max_df', 'max_features', 'binary', 'use_idf']:
            param_grid['features__text__{}'.format(i)] = tuning_params[i]

        # Handling other parameters is different depending on whether we will tune the model or not
        if self.is_model_tuning_active():
            param_grid['features__text__ngram_range'] = eval(' ,'.join([p for p in tuning_params["ngram_range"]]))
            param_grid['features__text__norm'] = [p for p in tuning_params["norm"] if p != 'None']
            if 'None' in tuning_params["norm"]:
                param_grid['features__text__norm'].append(None)
        else:
            param_grid['features__text__ngram_range'] = eval(tuning_params["ngram_range"])
            if tuning_params["norm"] == 'None':
                param_grid['features__text__norm'] = None
            else:
                param_grid['features__text__norm'] = tuning_params["norm"]
