from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from transformers import AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from get_trials import extract_protocol
modelf = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True)


    
# Fonction pour calculer la similarité cosinus entre le texte de réglementation d'un médicament et un autre texte donné
def cosineSimilarity(drug, countries, eudract, disease, text2): 
    # Extraction du texte de réglementation pour le médicament donné
    text1 =  extract_protocol(drug, countries, eudract, disease)
    
    # Création d'une liste contenant les deux textes (text1 et text2)
    documents = [text1, text2]
    
    # Initialisation du CountVectorizer pour convertir les textes en matrices de termes-document
    count_vectorizer = CountVectorizer(stop_words="english")
    
    # Conversion des textes en matrices de termes-document
    sparse_matrix = count_vectorizer.fit_transform(documents)

    # Conversion de la matrice sparse en une matrice dense
    doc_term_matrix = sparse_matrix.todense()
    
    # Création d'un DataFrame pour stocker les matrices de termes-document
    df1 = pd.DataFrame(
        doc_term_matrix,
        columns=count_vectorizer.get_feature_names_out(),
        index=["verite_terrain", "reglementation"],
    )

    # Calcul de la similarité cosinus entre les deux textes
    return cosine_similarity(df1, df1)


# Fonction pour prédire un score de "hallucination" entre le texte de réglementation d'un médicament et un autre texte donné
def score_hallucination(drug, countries, eudract, disease, text2):
    # Extraction du texte de réglementation pour le médicament donné
    text1 =  extract_protocol(drug, countries, eudract, disease)
    # Utilisation d'un modèle de prédiction (modelf) pour prédire un score en fournissant une liste contenant les textes text1 et text2
    scores = modelf.predict([
        [text1, text2]
    ])
    
    # Retourne le score prédit
    return scores


if __name__ == "__main__":
    filepath = "./data/2005-004866-17_AT.txt" #Nom du médicament Foscan, Substance active temoporfin
    
    # Lecture du contenu du fichier
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read() 
    print("cosineSimilarity: ", cosineSimilarity("temoporfin", "Austria - BASG", "2005-004866-17", "advanced head and neck squamous cell carcinoma", text))
    print("score_hallucination: ", score_hallucination("temoporfin", "Austria - BASG", "2005-004866-17", "advanced head and neck squamous cell carcinoma", text))
    
