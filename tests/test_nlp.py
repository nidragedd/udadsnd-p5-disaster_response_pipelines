"""
Unit tests class for NLP tasks

@author: nidragedd
"""
import unittest
import spacy

from src.models import transformers


class NLPTestCase(unittest.TestCase):

    def setUp(self):
        print('setUp NLP Test Case')
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

    def test_extract_verbs_from_text(self):
        text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of " \
               "the company took him seriously. “I can tell you very senior CEOs of major American car companies " \
               "would shake my hand and turn away because I wasn’t worth talking to,” said Thrun, in an interview " \
               "with Recode earlier this week."
        doc = self.nlp(text)
        expected = ['start', 'work', 'drive', 'take', 'can', 'tell', 'would', 'shake', 'turn', 'be', 'talk', 'say']
        self.assertEqual([token.lemma_ for token in doc if token.pos_ == "VERB"], expected)

    def test_lemmatize_text(self):
        text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of " \
               "the company took him seriously. “I can tell you very senior CEOs of major American car companies " \
               "would shake my hand and turn away because I wasn’t worth talking to,” said Thrun, in an interview " \
               "with Recode earlier this week."
        expected = 'sebastian thrun start work self drive car google people outside company take seriously tell ' \
                   'senior ceos major american car company shake hand turn away not worth talk say thrun interview ' \
                   'recode early week'
        self.assertEqual(transformers.lemmatize_txt(text), expected)

    def test_count_pos(self):
        text = '* How do beneficiaries (gender/economic status) in varying circumstances (emergency/non-emergency)' \
               ' spend cash?'
        doc = self.nlp(text)
        self.assertEqual(transformers.count_pos(doc, "VERB"), 3)
        self.assertEqual(transformers.count_pos(doc, "NOUN"), 7)
        self.assertEqual(transformers.count_pos(doc, "ADJ"), 2)


if __name__ == '__main__':
    unittest.main()
