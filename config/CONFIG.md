# PROJECT CONFIGURATION FILE
A default (and working) version of the `config.json` file is provided under the `config` directory in project's root directory.  
This file contains all parameters for this program and is divided into several sections, each one with a specific purpose:
* data
* database
* modeling

***WARNING***: please note that there is no (at least, not yet) automatic validation to ensure this file is well-formed so
if you provide another one or change things, you might break down the program. 

## 1. DATA
Only 3 parameters to specify where is the raw data and what are the CSV raw file names:
* `raw_input_folder`: please provide path to the directory where raw data are stored. If path is relative, please make 
sure that it will be relative to the folder that holds this config.json file
* `messages_file`: name of the messages raw file
* `categories_file`: name of the categories raw file

## 2. DATABASE
Only 3 parameters to specify where is (or will be stored) the SQLite database file, its name and the inner table name:
* `output_folder`: please provide path to the directory where SQLiteDB will be stored. If path is relative, please make 
sure that it will be relative to the folder that holds this config.json file
* `db_name`: name of the messages raw file
* `messages_table_name`: name of the main table within the database

## 3. MODELING
More tricky but not so hard neither. There are several levels, let's discover them one by one.

### 3.1. Main informations
* `output_folder`: please provide path to the directory where models will be stored (or loaded from). If path is relative, 
please make sure that it will be relative to the folder that holds this config.json file
* `available_models`: ***THIS MUST NOT BE CHANGED***, it is just informative. It is the list of available (so far) models
that you can decide to build/load/use. **Note that for the moment only Logistic Regression is available.**
* `model_to_build`: one of the values provided in `available_models` (so far only Logistic Regression is working)
* `tune_model`: boolean value to specify if we want to tune the models or not. See next section.

### 3.2. Tuning parameters
Depending on the `tune_model` parameter value, **we can build a model with default values or we can perform a grid search
cross validation** to find the best model.  
The first level corresponds to the elements that can be tuned. So far there is only the TF-IDF vectorizer but later we
could add also model parameters (for example Random Forest ones).  
In this section ***you are not allowed to change names*** (such as `tf-idf`, `default-values` or even the parameter names like
`ngram_range`, `max_df`, etc.), you can just change the values.  
Here is an example of such a configuration:
```
    "tuning_parameters": {
        "tf-idf": {
            "default_values": {
                "ngram_range": "(1, 2)",
                "max_df": 0.7,
                "max_features": 5000,
                "norm": "None",
                "binary": true,
                "use_idf": false
            },
            "gridsearch_values": {
                "ngram_range": ["(1, 1)", "(1, 2)"],
                "max_df": [0.5, 0.7, 1.0],
                "max_features": [3000, 5000],
                "norm": ["None", "l1", "l2"],
                "binary": [true, false],
                "use_idf": [true, false]
            }
        }
    }
```