import json
import pandas as pd
import plotly
from plotly.graph_objs import Bar
from flask import render_template, request, jsonify

from src import app

# index webpage displays cool visuals and receives user input text for model
from src.data import dataloader


@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')


@app.route('/about')
def about():
    return render_template('about.html')

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Get user query
    query = request.args.get('query', '')

    # Transform the given text into appropriate format to give to model and then use model to apply predict on query
    x_query = pd.Series([query], name='message')
    y_pred = dataloader.model.predict(x_query)[0]

    # Build a dictionnary where key is the human readable class name and value a binary 0/1 to specify if it belongs to
    # this class or not
    classification_results = dict(zip(dataloader.classes, y_pred))

    # This will render the go.html Please see that file.
    return render_template('go.html', query=query, classification_result=classification_results)


# Web page that will display some graphs about the training dataset
@app.route('/stats')
def plot():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = dataloader.df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres2',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('stats.html', ids=ids, graphJSON=graphJSON)
