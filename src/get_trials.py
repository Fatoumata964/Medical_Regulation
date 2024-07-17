import time
import uvicorn
from colabcode import ColabCode
from fastapi import FastAPI
from pydantic import BaseModel
from llm import llm
import pickle
import os
import re
import pandas as pd
import numpy as np
import sent2vec
from faiss_sim import faiss_search_similar_medications
from vector_store import vector_index
from llama_index.core.prompts.prompts import SimpleInputPrompt

app = FastAPI()
cc =  ColabCode(port=8002, code=False)
df = pd.read_csv("./data/df_final.csv")
llm = llm()

# Initialisation du modèle Sent2Vec
model = sent2vec.Sent2vecModel()
try:
      # Chargement du modèle depuis le chemin spécifié
        model.load_model("/content/drive/MyDrive/stage/hh/Medical_Reglementation/models/biosentvec.crdownload")
except Exception as e:
      # Gestion des erreurs lors du chargement du modèle
      print(e)



# Définition d'une classe TextInput héritant de BaseModel avec des champs par défaut pour le texte, les pays, le numéro EudraCT et la maladie
class TextInput(BaseModel):
    text: str = "irbesartan, hydrochlorothiazide"
    countries: str = "Belgium - FPS Health-DGM"
    eudract: str = "2004-000020-32"
    disease: str = None

# Fonction pour obtenir les réponses de LLM (Large Language Model) basées sur un dataframe de clusters, un prompt, un numéro EudraCT, un médicament et des pays
def get_llm(df_clus, prompt, eudract, drug, countries):
    # Création d'index vectoriels pour l'objectif principal, les objectifs secondaires, les critères d'inclusion et d'exclusion et les "primary endpoints" de l'essai clinique
    index_po, index_so, index_in, index_ex, index_ep = vector_index(df_clus)
    
    # Création de moteurs de requête pour chaque index
    query_engine_po = index_po.as_query_engine()
    query_engine_so = index_so.as_query_engine()
    query_engine_in = index_in.as_query_engine()
    query_engine_ex = index_ex.as_query_engine()
    query_engine_ep = index_ep.as_query_engine()

    # Descriptions textuelles pour différents objectifs et critères de l'essai clinique
    description_po = """L'objectif principal est l'objectif central de l'essai clinique. Il est souvent formulé comme une question de recherche claire et précise que l'essai vise à répondre. 
    Par exemple, dans un essai clinique sur un nouveau médicament pour le traitement du diabète, l'objectif principal pourrait être de démontrer que le médicament réduit significativement
    les niveaux de glucose dans le sang par rapport à un placebo."""
    description_so = """Les objectifs secondaires sont des buts supplémentaires de l'essai qui fournissent des informations complémentaires sur l'intervention étudiée. 
    Par exemple, dans le même essai clinique sur le médicament pour le diabète, des objectifs secondaires pourraient inclure l'évaluation de l'effet du médicament sur la réduction du poids, 
    les niveaux de cholestérol, ou la fréquence des événements hypoglycémiques."""
    description_in = """Les critères d'inclusion spécifient les caractéristiques que les participants doivent avoir pour être éligibles à participer à l'essai clinique."""
    description_ex = """Les critères d'exclusion spécifient les caractéristiques qui disqualifient des participants potentiels de l'essai clinique."""
    description_ep = """Les "primary endpoints" (points finaux primaires) sont les principales mesures de résultat d'un essai clinique.
    Ils sont utilisés pour évaluer directement l'objectif principal de l'étude. Ces points finaux sont définis avant le début de l'essai et jouent un rôle crucial dans la détermination de l'efficacité et/ou de la sécurité de l'intervention étudiée."""

    # Initialisation de la liste des réponses
    reponses = []

    # Exécution des requêtes pour chaque description et ajout des réponses à la liste
    response_po = query_engine_po.query(prompt + ' ' + description_po + ' ' + f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what is the Main objective of the trial?")
    reponses.append(response_po)
    response_so = query_engine_so.query(prompt + ' ' +  description_so + ' ' + f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what are the Secondary objectives of the trial?")
    reponses.append(response_so)
    response_in = query_engine_in.query(prompt + ' ' +  description_in + ' ' + f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what is the Principal inclusion criteria?")
    reponses.append(response_in)
    response_ex = query_engine_ex.query(prompt + ' ' +  description_ex + ' ' + f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what is the Principal exclusion criteria?")
    reponses.append(response_ex)
    response_ep = query_engine_ep.query(prompt + ' ' +  description_ep + ' ' + f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what is the Primary end point(s)?")
    reponses.append(response_ep)

    # Retourne la liste des réponses
    return reponses

def extract_protocol(drug, countries, eudract, disease):
    '''Extraction de l'essai clinique du médicament donné'''
    # Définition d'un prompt général
    prompt = """Tu es un assistant médical utile. Ton objectif est de donner des informations complétes sur les essais cliniques pharmaceutiques en étant le plus précis que possible 
    en fonction des instructions et du contexte fournis. La sortie doit contenir toute la réponse et chaque phrase doit commencer par une letter majuscule et terminer par un point."""

    #Si le médicament existe dans la base:
    if drug in df["Substance active"].values:

          # Récupérer les colonnes appartenant au protocole d'essai, elles commencent toutes par A.2.1 par exemple
          pattern = re.compile(r'^[A-Z]\..*')
          columns = [col for col in df.columns if pattern.match(col)]
          reponses = []
      
          for col in columns:
              # Extraire la description en supprimant la première partie avant le premier espace
              desc = ' '.join(col.split(' ')[1:])
              try:
                  # Récupérer la valeur pour la substance active donnée
                  value = df[col][df['Substance active'] == drug].iloc[0]
                  
                  text = f"{desc}: {str(value)}"
              except IndexError:
                  text = f"{desc}: Not Available"
              reponses.append(text)
          responses = '\n\n'.join(reponses)

    #Si le médicament n'existe pas dans la base:  
    else:
        # Intégration du texte du médicament et d'une phrase représentative de la maladie
        embedded_drug = model.embed_sentence(drug)
        embedded_disease = model.embed_sentence(str(disease))
        
        # Création de la matrice d'embedding en concaténant les embeddings du médicament et de la maladie
        embedding_mat = np.hstack((embedded_disease, embedded_drug))
        
        print("Drug embedded")
        
        # Chargement du modèle de clustering à partir du fichier pickle
        with open("./models/clustering_model.pkl", 'rb') as f:
            kmeans = pickle.load(f)
        # Prédiction du cluster auquel appartient le médicament
        y = kmeans.predict(embedding_mat)
        print(y)
        # Recherche de médicaments similaires dans le même cluster
        df_clus = df[df['cluster_labels'] == y[0]]
        sim = faiss_search_similar_medications(drug, disease, df_clus, 1)
        print(sim)
        # Récuperation des colonnes à générer avec l'IA générative
        similar_medications_in_cluster = (
          f"Main objective of the trial: " + sim['E.2.1 Main objective of the trial'].iloc[0] + '. ' 
          + f"Secondary objectives of the trial \n\n" + sim['E.2.2 Secondary objectives of the trial'].iloc[0] + '. ' 
          + f"Principal inclusion criteria: " + sim['E.3 Principal inclusion criteria'].iloc[0] + '. ' 
          + f"Principal exclusion criteria \n\n" + sim['E.4 Principal exclusion criteria'].iloc[0] + '. ' 
          + f"Primary end point: " + sim['E.5.1 Primary end point'].iloc[0]
            )
        print(similar_medications_in_cluster)
        # Ajout d'un promt particulier au cas où le médicament est inconnu
        prompt += f"""
            Si l'information demandée n’est pas spécifiée dans les informations contextuelles fournies, utilise l'exemple suivant pour répondre à la question :
            
            Exemple : {similar_medications_in_cluster}
            
            Réponds de manière complète et détaillée en utilisant l'exemple si nécessaire.
            """

        generative_titles = ["Main objective of the trial", "Secondary objectives of the trial", "Principal inclusion criteria", "Principal exclusion criteria", "Primary end point(s)"]
      # Extraction des parties restantes du protocole sans IA générative
      # Pattern pour les colonnes
        pattern = re.compile(r'^[A-Z]\..*')
        columns = [col for col in df.columns if pattern.match(col)]
      
      # Stockage des réponses
        responses1 = []
      
      # Initialisation de la partie générative
        generative_responses = get_llm(sim, prompt, eudract, drug, countries)
      
      # Dictionnaire pour stocker les réponses génératives
        generative_dict = dict(zip(generative_titles, generative_responses))
      
        for col in columns:
          # Extraire la description en supprimant la première partie avant le premier espace
            desc = ' '.join(col.split(' ')[1:])
          
          # Vérifier si la colonne est dans la liste des titres pour utiliser l'IA générative
            if desc in generative_titles:
                value = generative_dict[desc]
            else:
                try:
                    value = sim[col].iloc[0]
                  # Vérifier et traiter si c'est une liste même après transformation
                    if isinstance(value, list):
                        value = ', '.join(value)
                except IndexError:
                    value = "Not Available"
          
          # Ajouter la réponse formatée à la liste des réponses
            text = f"{desc}: {str(value)}"
            responses1.append(text)
      
      # Joindre les réponses
        response = '\n\n'.join(responses1)
        for el in [sim['Substance active'].iloc[0], sim['Nom du médicament'].iloc[0]]:
              pattern = re.compile(re.escape(el), re.IGNORECASE)
              response = pattern.sub(drug, response)
                    
        responses = f"CECI EST UN EXEMPLE D'ESSAI CLINIQUE PROCHE DE CELUI DEMANDE.\n\n{response}"
        

    return responses
    
@app.get('/')
def index():
    """
    Default endpoint for the API.
    """
    return {
        "version": "0.1.0",
        "documentation": "/docs"
    }

# Define a route that accepts POST requests with JSON data containing the text
@app.post("/apiv1/protocol/get-protocol")
async def get_protocol(drug: TextInput):
    # You can perform text processing here
    start_time = time.time()
    regulation_text = extract_prorocol(drug.text.lower(), drug.countries.lower(), drug.eudract.lower(), drug.disease.lower())
    # stopping the timer
    stop_time = time.time()
    elapsed_time = stop_time - start_time

    # formatting the elapsed time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(regulation_text)
    # Return the processed text as a response
    return {"Protocol d'essai clinique": regulation_text}


if __name__ == '__main__':
    cc.run_app(app=app)
