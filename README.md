# Udacity DataScience nanodegree project - Disaster Response Pipeline Project
## 1. CONTEXT - OBJECTIVES
### Context
This project is only for educational purposes. I did it while I was following the Udacity `DataScience nanodegree`.  
Machine learning is critical to helping different organizations understand which messages are relevant to them and which
messages to prioritize.  
During these disasters is when they have the least capacity to filter out messages that matter, and find basic methods 
such as using key word searches to provide trivial results.

### Objectives
In this project our goal is to analyze thousands of real messages provided by [Figure 8](https://www.figure-eight.com/),
sent during natural disasters either via social media or directly to disaster response organizations.  
We have several steps to follow:  
1.**Build an ETL pipeline** that processes message and category data from csv files and load them into a SQLite database
2. **Build a Machine Learning pipeline** that will then read from SQLite DB to create and save a multi-output supervised 
learning model (yes, this is multi-class classification problem). Goal is to categorize these events so that we can send
the messages to an appropriate disaster relief agency.
3. **Build a webapp** that will:
    * be able to first launch both pipelines in order to populate everything that needs to be
    * provide some data visualizations
    * use our trained and saved model to classify new messages for 36 categories.

This webapp will be used by an emergency worker: he gives a new message and gets classification results in several
categories.

---
## 2. ABOUT THE DATA
### Global overview
**TODO** + images

### Imbalanced data: drawbacks
In the given dataset, data is imbalanced. It means that some of the categories appears a lot whereas some others appears
much less often. This is an issue in Machine Learning because we have too few examples to learn well and if our model
often see the same value it will obviously 'learn' that and tend to predict this value.  
**That is why we have to wisely choose the performance metric!**.  

#### Which performance metric?
If we choose `accuracy` as the performance metric to check whether we classify well or not, a dummy classifier that always
predict the most frequent class will have a good score but in the end we will have built a very poor model.  
When data are imbalanced we can use other metrics such as:
* ***Precision:*** among all the positive predictions made by the model, we count how many of them were actually positive
in the train/validation/test dataset. This is a ratio and the higher the better because it means that our model is very
precise. In other words, when the model says it is True, it is actually True (in this case there are few _"False
Positives"_).
* ***Recall:*** among all the real positive values in the train/validation/test dataset, how many of them did our model
classified as positive? This metric indicates how good is our model to "catch them all". Indeed, the model can be very
precise (when it says it is positive, it is correct) but could still miss a lot of positive samples. And this is not 
good neither. Recall is also a ratio and the higher the better (in this case there are few _"False Negatives"_).

Depending on the use case, we might want to focus on `Recall` rather than `Precision` or vice-versa.  
For example in medical domain, when algorithm says the person has/does not have cancer, we want this information to be as
accurate as possible (you can easily imagine the impacts when this is wrong), this is the _Precision_.  
But we also want to avoid saying someone that he does not have cancer whereas he actually has one (this is the worst 
case scenario). So perhaps in this case we want to focus more on _Recall_ (ensure we catch all of them) even if it means
that it is less precise.

If we do not want to choose between _Precision_ or _Recall_ because both are kind of equally important we can choose to
use the ***F1-Score*** metric which is the harmonic mean of both:  
F1 score = 2x ((Precision x Recall)/(Precision + Recall))

**This will be the selected metric in our case.**

Here are few interesting readings about performance metrics in classification:
* [Accuracy, Recall, Precision, F-Score & Specificity, which to optimize on?](https://towardsdatascience.com/accuracy-recall-precision-f-score-specificity-which-to-optimize-on-867d3f11124)
* [How data scientists can convince doctors that AI works](https://towardsdatascience.com/how-data-scientists-can-convince-doctors-that-ai-works-c27121432ccd)


#### Ensure that we have enough data to train on
When data are imbalanced there is also another issue: how to ensure that in our training dataset we will have enough
samples of each class to train on and then be able to learn and classify correctly?  
The split between train and test dataset cannot then be totally random. There is this *[scikit-multilearn package](http://scikit.ml/index.html)*
that can help to [stratify the data](http://scikit.ml/stratification.html).  
We can also use other techniques such as:
* ***oversampling***: here we duplicate the data for classes that appears less so that in the end there are more of them and
algorithm can learn. The drawback is that depending on the use case, as it is the same data that is duplicated, the 
learning can have a big bias.
* ***undersampling***: randomly remove some occurrences of classes that appears the most. This is also a good idea but it can
lead us to a lot of data loss depending on how huge is the gap between classes.
* we could mix both oversampling and undersampling

***Note:*** none of this technique has been applied to this project, this could be a further improvement.

This is algo a good reading about metrics and resampling: [Dealing with Imbalanced Data](https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18)

### Modeling
As it is a supervised classification problem I will take the **Logistic Regression* algorithm as a baseline and it will be
compared to tree based algorithms which are known to handle pretty well imbalanced data.

---
## 3. WEBAPP SCREENSHOTS

**TODO**



---
## 4. TECHNICAL PART
### Dependencies & Installation - Create your CONDA virtual environment
Easiest way is to create a virtual environment through **[conda](https://docs.conda.io/en/latest/)**
and the given `environment.yml` file by running this command in a terminal (if you have conda, obviously):
```
conda env create -f environment.yml
```

If you do not have/want to use conda for any reason, you can still setup your environment by running some `pip install`
commands. Please refer to the `environment.yml` file to see what are the dependencies you will need to install.  
Basically, this project requires **Python 3.7** in addition to common datascience packages (such as 
[numpy](https://www.numpy.org/), [pandas](https://pandas.pydata.org/), 
[sklearn](https://scikit-learn.org/stable/), [matplotlib](https://matplotlib.org/), 
[seaborn](https://seaborn.pydata.org/) and so on).

For modeling, this project is using **TODO**.  

There are those additional packages in order to expose our work within a webapp:
**TODO !!!**
* [Flask](https://radimrehurek.com/gensim/): used for topic modeling
* [plotly](https://pypi.org/project/wordcloud/): used to generate some tag clouds



---
### Directory & code structure
Here is the structure of the project:
```
    project
      |__ assets    (contains images displayed in notebooks)
      |__ config    (configuration section, so far it contains JSON files used to configure logging or the program)
      |__ data      (raw data)
            |__ output  (data processed and/or database saved locally)
      |__ notebooks  (contains all notebooks)
      |__ src       (python modules and scripts)
            |__ config  (scripts called to actually configure the program)
            |__ models  (python scripts used to build, train and save a ML model)
            |__ preprocessing  (python scripts used to preprocess the raw data)
            |__ webapp  (python scripts and HTML templates used to render the webapp)
```

### Run the app on your local computer
**TODO**
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
