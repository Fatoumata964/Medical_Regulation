import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss



# Fonction pour rechercher des médicaments similaires à un médicament donné à l'aide de l'index FAISS
def faiss_search_similar_medications(subs, disease, df, k):

    df['Diseases_str'] = df['Diseases'].apply(lambda x: ' '.join(x))
    df['combined_text'] = df['Substance active'] + ' ' + df['Diseases_str']
    df['combined_text'] = df['combined_text'].astype(str)

    tfidf_vectorizer = TfidfVectorizer()
    text_vectors = tfidf_vectorizer.fit_transform(df['combined_text']).toarray()
    text_vectors = text_vectors.astype('float32')


    index = faiss.IndexFlatL2(text_vectors.shape[1])
    index.add(text_vectors)

    query = subs + ' ' + disease

    _vector = tfidf_vectorizer.transform([query]).toarray()
    _vector = _vector.astype('float32')

    query_vector = _vector.reshape(1, -1)
    distances, indices = index.search(query_vector, k)


    similar_drugs = df.iloc[indices[0]]

    return similar_drugs
