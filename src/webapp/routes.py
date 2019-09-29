import json
import pandas as pd
import plotly
from plotly.graph_objs import Bar, Pie
from flask import render_template, request

from src import app
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
    # Extract all data needed for graphs
    genre_counts = dataloader.df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    related_genre_counts = dataloader.df[dataloader.df['related'] == 1]['genre'].value_counts()
    not_related_genre_counts = dataloader.df[dataloader.df['related'] == 0]['genre'].value_counts()

    per_category_counts = dataloader.df.drop(['id', 'message', 'genre'], axis=1).sum()
    categories = dataloader.df.drop(['id', 'message', 'genre'], axis=1).columns

    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Nb messages"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(name='related', x=genre_names, y=related_genre_counts),
                Bar(name='not related', x=genre_names, y=not_related_genre_counts)
            ],
            'layout': {
                'title': 'Genre of messages vs. "Related"',
                'yaxis': {'title': "Nb messages"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Pie(labels=categories, values=per_category_counts)
            ],
            'layout': {
                'title': 'Nb messages per category (in pie chart)',
                'yaxis': {'title': "Nb messages"},
                'xaxis': {'title': "Category"}
            }
        },
        {
            'data': [
                Bar(x=categories, y=per_category_counts)
            ],
            'layout': {
                'title': 'Nb messages per category',
                'yaxis': {'title': "Nb messages"},
                'xaxis': {'title': "Category"}
            }
        }

    ]
    # Create visuals

    # Encode graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render template with elements to display graphs
    return render_template('stats.html', ids=ids, graphJSON=graphJSON)
