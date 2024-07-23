import os
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import PreProcessor

folder_path = 'dataset'
lyric_col = 'lyric_en'
preprocessor = PreProcessor()
dfs = []
top_n = 20

def remove_non_alphanumeric(input_string):
    pattern = r'[^a-zA-ZáéíóúàèìòùãõâêîôûçÁÉÍÓÚÀÈÌÒÙÃÕÂÊÎÔÛÇ\s]'
    cleaned_string = re.sub(pattern, ' ', input_string)
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string).strip()
    cleaned_string = cleaned_string.replace('  ', ' ')
    return cleaned_string.lower()

def concat_lyrics(lyric):
    new_lyric = ''.join(str(lyric))
    #new_lyric = remove_non_alphanumeric(new_lyric)
    return new_lyric

def remove_duplicates(lyric):
    unique_words = []
    for word in lyric.split():
        if str(word.lower()) not in unique_words:
            unique_words.append(str(word.lower()))
    response = ' '.join(unique_words)
    return response

def preprocess(lyric, should_remove_duplicates=False):
    new_lyric = str(lyric)
    
    if should_remove_duplicates:
        new_lyric = remove_duplicates(lyric)
    new_lyric = preprocessor.preprocess(new_lyric)
    return new_lyric

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        genre = filename.split('lyrics_')[1].split('.csv')[0]
        df = pd.read_csv(file_path)
        df['genre'] = genre
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

dfs = df.groupby('genre')[lyric_col].apply(concat_lyrics).reset_index()
dfs[lyric_col] = dfs[lyric_col].apply(preprocess)

documents = dfs[lyric_col]

# Inicializa o vetorizador TF-IDF
vectorizer = TfidfVectorizer()

# Ajusta e transforma os documentos em uma matriz TF-IDF
tfidf_matrix = vectorizer.fit_transform(documents)

# Obtem os nomes das palavras (features)
feature_names = vectorizer.get_feature_names_out()

# Função para obter os top N termos de um documento
def get_top_n_terms(tfidf_vector, feature_names, n=top_n):
    sorted_indices = np.argsort(tfidf_vector.toarray()).flatten()[::-1]
    top_n_indices = sorted_indices[:n]
    top_terms = [(feature_names[i], tfidf_vector[0, i]) for i in top_n_indices]
    return top_terms

# Exibe os termos mais importantes para cada documento
for doc_idx, tfidf_vector in enumerate(tfidf_matrix):
    top_terms = get_top_n_terms(tfidf_vector, feature_names, top_n)
    print(f"\n{dfs['genre'][doc_idx]}")
    for term, value in top_terms:
        print(f"  Termo: {term}, TF-IDF: {value:.4f}")