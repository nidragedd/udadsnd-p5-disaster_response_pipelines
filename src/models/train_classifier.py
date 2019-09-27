"""
Python script to call to launch model building steps:
    1. Load messages and categories from DB (execute first the preprocessing.process_data script if not exists!)
    2. Split dataset into training and test sets
    3. Build a text processing and machine learning pipeline
    4. Train and tune (optional) a model using GridSearchCV
    5. Output results on the test set
    6. Export the final model as a pickle file

@author: nidragedd
"""
import logging
# Used to serialize the models on disk
from sklearn.externals import joblib

from sqlalchemy import create_engine
import pandas as pd

# NLP transformations
from sklearn.feature_extraction.text import TfidfVectorizer
# sklearn pipelines import
from sklearn.pipeline import Pipeline, FeatureUnion

# Training
from sklearn.model_selection import train_test_split, GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Model evaluation
from sklearn.metrics import classification_report, f1_score, make_scorer, precision_score, recall_score

import sys
import os

# This hack is mandatory in order to be able to import modules from sibling packages
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from src.config import utils
from src.config.pgconf import ProgramConfiguration
from src.data import dataloader
from src.models import transformers


def load_data():
    """
    Load the data from local SQLite database and return X and Y (i.e features and targets)
    :return: X and Y (i.e features and targets)
    """
    db_file = utils.pgconf.get_database_file()
    logger.info("Loading data from SQLite DB '{}'".format(db_file))
    engine = create_engine('sqlite:///{}'.format(db_file))

    df = pd.read_sql_table(utils.pgconf.get_db_table_name(), engine)
    if df is not None:
        if df.shape[0] > 0:
            logger.info("Control check: {} rows read in dataframe".format(df.shape[0]))

            X = df['message']
            Y = df.drop(['id', 'message', 'genre'], axis=1)
            return X, Y, Y.columns.to_list()
        else:
            raise Exception("DB file '{}' is empty".format(db_file))
    else:
        raise Exception("Could not read SQLite DB file '{}'".format(db_file))


def build_model():
    """
    Build the model. So far there is only one model available: logistic regression but a further improvement would be to
    let the user choose one model among several models (this would be done within the external JSON configuration file)
    :return: (sklearn pipeline) the pipeline object to fit_transform against training data
    """
    logger.info("Building the model")
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("text", TfidfVectorizer(tokenizer=transformers.lemmatize_txt)),
            ("word-count", transformers.WordCountExtractor()),
            ("word-len", transformers.AverageWordLengthExtractor()),
            ("sentence-count", transformers.SentenceCountExtractor()),
            ("sentence-len", transformers.AverageSentenceLengthExtractor()),
            ("verb-count", transformers.PosCountExtractor())
        ])),
        ("clf", OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000)))
    ])

    # Handle TF-IDF parameters (values come from external JSON configuration file)
    param_grid = {}
    utils.pgconf.handle_tfidf_tuning_param_grid(param_grid)

    # If user choose to tune the model then we wrap our pipeline within a cross validation with GridSearchCV
    if utils.pgconf.is_model_tuning_active():
        logger.info("Tuning has been set to True within configuration file, using GridSearchCV!")
        pipeline = GridSearchCV(pipeline, param_grid, cv=3, scoring=make_scorer(f1_score, average='micro'), verbose=3)
    else:
        logger.info("Tuning has been set to False within configuration file, using default given values")
        # Need to find a more beautiful way to handle this ugly part (will see that later)
        pipeline.set_params(features__text__ngram_range=param_grid['features__text__ngram_range'])
        pipeline.set_params(features__text__max_df=param_grid['features__text__max_df'])
        pipeline.set_params(features__text__max_features=param_grid['features__text__max_features'])
        pipeline.set_params(features__text__norm=param_grid['features__text__norm'])
        pipeline.set_params(features__text__binary=param_grid['features__text__binary'])
        pipeline.set_params(features__text__use_idf=param_grid['features__text__use_idf'])

    logger.info("Parameters are: {}".format(param_grid))
    logger.info("Pipeline that will be used is {}".format(pipeline))

    return pipeline


def evaluate_model(model_name, pipeline, x_test, y_test):
    """
    Evaluate model performance by predicting on test dataset and then printing performance metrics
    :param model_name: (string) name of the model, just for display purpose
    :param pipeline: (object) the sklearn classifier pipeline to use for inference
    :param x_test: (pandas DataFrame) features of the test subset used to evaluate
    :param y_test: (pandas DataFrame) ground truth targets for the test subset used to evaluate
    """
    y_pred = pipeline.predict(x_test)

    p = precision_score(y_test, y_pred, average="micro")
    r = recall_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")
    wp = precision_score(y_test, y_pred, average="weighted")
    wr = recall_score(y_test, y_pred, average="weighted")
    wf1 = f1_score(y_test, y_pred, average="weighted")

    logger.info("Model {} -- MICRO metrics -- Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".
                format(model_name, p, r, f1))
    logger.info("Model {} -- WEIGHTED metrics -- Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".
                format(model_name, wp, wr, wf1))

    # Build a df with results, class per class
    logger.info("Classification report below:\n{}".format(classification_report(y_test, y_pred)))


def save_model(model_name, model, classes):
    """
    Save the model locally
    :param model_name: (string) just for display purpose
    :param model: (sklearn object) the model to dump
    :param classes: (array) list of classes
    """
    model_save_file = utils.pgconf.get_output_model_file()
    classes_file = utils.pgconf.get_output_model_classes_file()
    logger.info("Saving {} model to '{}'".format(model_name, model_save_file))
    joblib.dump(model, model_save_file)
    joblib.dump(classes, classes_file)
    logger.info("Model saved!")


def train():
    """
    Training process: load data, build a model, train, evaluate then save on disk
    """
    X, Y, classes = load_data()
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model_name = utils.pgconf.get_chosen_model_name()
    model = build_model()

    logger.info("Fit {} model with training data".format(model_name))
    model.fit(x_train_val, y_train_val)
    logger.info("{} model trained".format(model_name))

    evaluate_model(model_name, model, x_test, y_test)

    save_model(model_name, model, classes)

    # Coherence check
    saved_model, classes = dataloader.load_model_and_classes()
    evaluate_model('{} loaded from disk'.format(model_name), saved_model, x_test, y_test)


logger = logging.getLogger()

if __name__ == '__main__':
    # Handle mandatory arguments
    args = utils.parse_args()
    config_file = vars(args)['config_file']
    log_config_file = vars(args)['log_file']

    # Configure the whole program (logging, external config files, singletons, ...)
    utils.configure_logging(log_config_file)
    utils.pgconf = ProgramConfiguration(config_file)

    train()
