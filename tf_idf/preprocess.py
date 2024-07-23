import nltk
import re

import numpy as np

from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class PreProcessor:
    def preprocess(self, documents):
        return self.preprocess_document(documents)

    def preprocess_document(self, document):
        try:
            preprocessed = document.lower()
            preprocessed = self.remove_chords(preprocessed)
            preprocessed = self.remove_stopword(preprocessed)
            #preprocessed = self.pos_tagger(preprocessed)
            #preprocessed = self.stemm(preprocessed)
            return ' '.join(preprocessed)
        except Exception as err:
            print(err)
            return []

    def remove_chords(self, document):
        pattern = r'\b[A-Ga-g][#b]?[mM]?\d*\b'
        cleaned_text = re.sub(pattern, '', document)
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
        return cleaned_text

    def remove_stopword(self, document):
        stop_words = nltk.corpus.stopwords.words("english")
        custom_stop_words = ['nan', 'uou', 'ig', 'ei', 'nã', 'vampiro', 'aê', 'pa', 'lo', 'oh', 'sh', 'gonzaga', 'yuri', 'si']
        extended_stop_words = np.concatenate([stop_words, custom_stop_words])
        return [
            word for word in simple_preprocess(str(document)) if word not in extended_stop_words
        ]

    def stemm(self, document):
        ps = PorterStemmer()
        return [ps.stem(w) for w in document]

    def pos_tagger(self, document):
        tagged = nltk.pos_tag(document)
        allowed_pos_tag = [
            "JJ",
            "NN",
            "VB",
        ]
        return [item[0] for item in tagged if item[1] in allowed_pos_tag]
