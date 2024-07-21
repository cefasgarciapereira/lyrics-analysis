import re
import sys
import gensim
import nltk
import pyLDAvis

import numpy as np
import pandas as pd
import gensim.corpora as corpora
import pyLDAvis.gensim_models as gensimvis

from pprint import pprint
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from preprocess import PreProcessor

pathname = sys.argv[1]
num_topics = 5

nltk.download('stopwords')
nltk.download("averaged_perceptron_tagger")

preprocessor = PreProcessor()

df = pd.read_csv(pathname)

print(df.head(3))

documents = df['lyric_en']

processed_texts = preprocessor.preprocess(documents)
texts = processed_texts
id2word = corpora.Dictionary(processed_texts)
corpus = [id2word.doc2bow(text) for text in texts]
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, passes=20, per_word_topics=True, alpha=0.3, eta=1)
pprint(lda_model.print_topics())

coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score:', coherence_lda)

p = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(p, f'lda_{pathname.split(".csv")[0].split("lyrics_")[1]}.html')