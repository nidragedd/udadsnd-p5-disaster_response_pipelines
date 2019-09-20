"""
Flask webapp main file
Steps:
    1. Load external configuration files (logging and program configuration)
    2. Load data from DB
    3. Start the Flask app server

@author: nidragedd
"""
import logging

from src import app
from src.config import utils
from src.config.pgconf import ProgramConfiguration
from src.data import dataloader
#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize

#from sklearn.externals import joblib



# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#
#     return clean_tokens
#
# # load model
# model = joblib.load("../models/your_model_name.pkl")

logger = logging.getLogger()
df = None

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

    if ok:
        # Launch the Flask app server
        app.run(host='0.0.0.0', port=3001, debug=True)
