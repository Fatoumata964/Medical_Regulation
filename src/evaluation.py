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
    texte = extract_protocol(drug, countries, eudract, disease)
    regex = r"""
          (E\.2\.1\s.*?(?=\nE\.2\.2|E\.3|E\.4|E\.5|E\.6|$))|
          (E\.2\.2\s.*?(?=\nE\.2\.3|E\.3|E\.4|E\.5|E\.6|$))|
          (E\.3(\.\d+)?\s.*?(?=\nE\.4|E\.5|E\.6|$))|
          (E\.4(\.\d+)?\s.*?(?=\nE\.5|E\.6|$))|
          (E\.5\.1\s.*?$)
          """

    # Compilation de l'expression régulière avec les flags re.VERBOSE et re.DOTALL
    pattern = re.compile(regex, re.VERBOSE | re.DOTALL)
    
    # Extraction des sections spécifiques
    matches = pattern.findall(texte)
    
    # Filtrage pour retirer les groupes vides et les lignes non désirées
    filtered_matches = [match for match_tuple in matches for match in match_tuple if match]

    text1 = ""
    # Affichage des résultats
    for match in filtered_matches:
        text1 += match.strip()  + "\n"
    
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
    texte = extract_protocol(drug, countries, eudract, disease)
    regex = r"""
          (E\.2\.1\s.*?(?=\nE\.2\.2|E\.3|E\.4|E\.5|E\.6|$))|
          (E\.2\.2\s.*?(?=\nE\.2\.3|E\.3|E\.4|E\.5|E\.6|$))|
          (E\.3(\.\d+)?\s.*?(?=\nE\.4|E\.5|E\.6|$))|
          (E\.4(\.\d+)?\s.*?(?=\nE\.5|E\.6|$))|
          (E\.5\.1\s.*?$)
          """

    # Compilation de l'expression régulière avec les flags re.VERBOSE et re.DOTALL
    pattern = re.compile(regex, re.VERBOSE | re.DOTALL)
    
    # Extraction des sections spécifiques
    matches = pattern.findall(texte)
    
    # Filtrage pour retirer les groupes vides et les lignes non désirées
    filtered_matches = [match for match_tuple in matches for match in match_tuple if match]

    text1 = ""
    # Affichage des résultats
    for match in filtered_matches:
        text1 += match.strip()  + "\n"
    # Utilisation d'un modèle de prédiction (modelf) pour prédire un score en fournissant une liste contenant les textes text1 et text2
    scores = modelf.predict([
        [text1, text2]
    ])
    
    # Retourne le score prédit
    return scores


if __name__ == "__main__":
    text = """ E.2.1 Main objective of the trial: To assess efficacy and safety of FoscanB (temoporfiin) photodynamic therapy in the treatment of locally advanced hilar or extrahepatic bile duct carcinoma without distant metastases.
          E.2.2 Secondary objectives of the trial: - Progression-free survival time, overall survival time
          - Rate of systemic response (RECIST criteria / EORTC)(37).
          - Toxicity using WHO criteria and criteria for local toxicity in the biliary system
          E.3 Principal inclusion criteria: * bile duct carcinoma proven by histology in advanced or non-operable stage or tumor extension:
          a) Bismuth type 111 or IV ( not resectable with RO-margins ),
          b) Bismuth type I or II, if resective surgery is contraindicated for old age
          or poor surgical risk of patient
          * sufficient general condition to undergo PDT (Karnofsky status 2 30%)
          * age> 19 years
          * access to common bile duct (either via endoscopy after sphincterotomy or percutaneously after transhepatic drainage),
          * informed written consent for PDT.
          E.4 Principal exclusion criteria: * porphyria or other diseases exacerbated by light
          * known intolerance or allergies to porphyrin derivatives
          * a planned surgical procedure within the next 30 days
          * coexisting ophthalmic disease likely to require slit lamp examination within the next 30 days
          * impaired kidney or liver function (creatinine > 2.5x elevated, INR > 2.2 on vitamin K),
          * leukopenia ( WBC < 2000/cmm ) or thrombopenia ( < 50000/cmm ),
          * cytotoxic chemotherapy within the past 4 weeks.
          * pregnancy ( and safe contraception for 6 months after PDT )
          * accompaning/complicating disease with very poor prognosis (expected survival < 6 weeks),
          * proven advanced peritoneal carcinomatosis (PET scan imaging, ascites positive for tumor cells)
          
          E.5.1 Primary end point(s): Rate of local response and depth of tumoricidal tissue penetration of Foscan-PDT

    """
    print("cosineSimilarity: ", cosineSimilarity("Foscan", "Austria - BASG", "2005-004866-17", "advanced head and neck squamous cell carcinoma", text))
    print("score_hallucination: ", score_hallucination("Foscan", "Austria - BASG", "2005-004866-17", "advanced head and neck squamous cell carcinoma", text))
    
