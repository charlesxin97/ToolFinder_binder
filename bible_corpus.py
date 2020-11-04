from nltk.corpus import gutenberg
from string import punctuation
from gensim.models import word2vec
import nltk
import numpy as np
import re

class get_bible():
    def __init__(self):
        self.wpt = nltk.WordPunctTokenizer()
        self.stop_words = nltk.corpus.stopwords.words('english')

    def normalize_document(self,doc):
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
        doc = doc.lower()
        doc = doc.strip()
        # tokenize document
        tokens = self.wpt.tokenize(doc)
        # filter stopwords out of document
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        # re-create document from filtered tokens
        doc = ' '.join(filtered_tokens)
        return doc
    def get_corpus(self):

        normalize_corpus = np.vectorize(self.normalize_document)
        bible = gutenberg.sents('bible-kjv.txt')
        remove_terms = punctuation + '0123456789'

        norm_bible = [[word.lower() for word in sent if word not in remove_terms] for sent in bible]
        norm_bible = [' '.join(tok_sent) for tok_sent in norm_bible]
        norm_bible = filter(None, normalize_corpus(norm_bible))
        norm_bible = [tok_sent for tok_sent in norm_bible if len(tok_sent.split()) > 2]
        return norm_bible