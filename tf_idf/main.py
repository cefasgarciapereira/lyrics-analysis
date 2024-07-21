from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Exemplo de 10 documentos
documents = [
    "O gato está no tapete.",
    "O gato gosta de peixe.",
    "O tapete é bonito.",
    "Cachorros são legais.",
    "O peixe nada no aquário.",
    "Gatos gostam de tapetes.",
    "Tapetes são confortáveis.",
    "Aquários precisam de limpeza.",
    "Peixes são silenciosos.",
    "Limpeza de aquários é importante."
]

# Inicializa o vetorizador TF-IDF
vectorizer = TfidfVectorizer()

# Ajusta e transforma os documentos em uma matriz TF-IDF
tfidf_matrix = vectorizer.fit_transform(documents)

# Obtem os nomes das palavras (features)
feature_names = vectorizer.get_feature_names_out()

# Função para obter os top N termos de um documento
def get_top_n_terms(tfidf_vector, feature_names, n=5):
    sorted_indices = np.argsort(tfidf_vector.toarray()).flatten()[::-1]
    top_n_indices = sorted_indices[:n]
    top_terms = [(feature_names[i], tfidf_vector[0, i]) for i in top_n_indices]
    return top_terms

# Número de termos mais importantes para exibir
top_n = 5

# Exibe os termos mais importantes para cada documento
for doc_idx, tfidf_vector in enumerate(tfidf_matrix):
    top_terms = get_top_n_terms(tfidf_vector, feature_names, top_n)
    print(f"Documento {doc_idx + 1}:")
    for term, value in top_terms:
        print(f"  Termo: {term}, TF-IDF: {value:.4f}")
    print()
