"""
Module that holds all custom transformers to use within models pipelines

@author: nidragedd
"""
import pandas as pd
import numpy as np
import spacy
import re

from sklearn.base import BaseEstimator, TransformerMixin


# Dependency labels, Name Entity Recognition and Text Categorization are not need for our specific usage
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])


class WordCountExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe and outputs number of words"""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def nb_words(self, text):
        return len(text.split())

    def transform(self, x):
        """
        Need to encapsulate the result within a DataFrame otherwise an error is thrown ("ValueError: blocks[0,:] has
        incompatible row dimensions. Got blocks[0,1].shape[0] == 1, expected 20972.")
        """
        return pd.DataFrame(x.apply(self.nb_words))


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe and outputs average word length"""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def average_word_length(self, text):
        """
        Sometimes text is empty so need to handle this case
        """
        return np.mean([len(word) for word in text.split()]) if len(text.split()) > 0 else 0

    def transform(self, x):
        return pd.DataFrame(x.apply(self.average_word_length))


class SentenceCountExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe and outputs number of sentences"""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def nb_sentences(self, text):
        return len(get_sentences(text))

    def transform(self, x):
        return pd.DataFrame(x.apply(self.nb_sentences))


class AverageSentenceLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe and outputs average sentence length"""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def average_sentence_length(self, text):
        return np.mean([len(sentence) for sentence in get_sentences(text)]) if len(get_sentences(text)) > 0 else 0

    def transform(self, x):
        return pd.DataFrame(x.apply(self.average_sentence_length))


class PosCountExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe and outputs number of words tagged as the given part-of-speech (POS)"""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def nb_pos(self, text):
        """
        For performance reasons, we load through spacy nlp only once and get all counts we are interested in
        """
        doc = nlp(text)
        nb_nouns = count_pos(doc, "NOUN")
        nb_verbs = count_pos(doc, "VERB")
        nb_adjectives = count_pos(doc, "ADJ")
        return nb_nouns, nb_verbs, nb_adjectives

    def transform(self, x):
        df = pd.DataFrame(x.apply(self.nb_pos))
        # At this point all we have is a dataframe with only one column where each value is a tuple (nb_nouns, nb_verbs,
        # nb_adjectives)
        # We have to split and transform into 3 columns
        df = df.astype(str).message.str[1:-1].str.split(',', expand=True)
        # Do not forget to put back all values as numeric ones
        return df.astype(int)


def lemmatize_txt(x):
    """
    Lemmatize text from a string.
    :param x: (string) the text we want to lemmatize
    :return: (string) string containing the lemmatized text.
    """
    doc = nlp.tokenizer(x.lower())
    lemma_txt = [token.lemma_ for token in doc
                 if not token.is_punct | token.is_space | token.is_stop | token.is_digit | token.is_quote |
                        token.is_bracket | token.is_currency | token.like_url | token.like_email]
    lemma_txt = ' '.join(lemma_txt)
    return lemma_txt


def count_pos(doc, pos):
    """
    Count how many elements corresponding to the given part-of-speech appears in a given document
    :param doc: (string) the text to analyze
    :param pos: (string) part-of-speech spacy tag
    :return: (int) the number of elements corresponding to the given part-of-speech appears in a given document
    """
    return len([token for token in doc if token.pos_ == pos])


def get_sentences(text):
    """
    Retrieve all sentences within a given text. I have decided that sentences have words and several characters and
    should end by either a '.', a '?' or a '!'
    :param text: (string) the text to analyze
    :return: (array) the detected sentences
    """
    # Add a trailing dot if the last character is not ".", "?" or "!" to ensure we will capture the last sentence
    if text.strip()[-1:] not in ['.', '?', '!']:
        text = text + '.'
    sentence_detection_regex = r"\s?[\w,-;:'()\"\s]+[.?!]"
    sentences = re.findall(sentence_detection_regex, text)
    return sentences
