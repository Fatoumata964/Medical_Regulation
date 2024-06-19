import time
import uvicorn
from colabcode import ColabCode
from fastapi import FastAPI
from pydantic import BaseModel
from llm import llm
import pickle
import os
import pandas as pd
import numpy as np
import sent2vec
from vector_store import vector_index
from llama_index.core.prompts.prompts import SimpleInputPrompt

app = FastAPI()
cc =  ColabCode(port=8002, code=False)
df = pd.read_csv("./data/df_final.csv")

# Initialisation du modèle Sent2Vec
model = sent2vec.Sent2vecModel()
try:
      # Chargement du modèle depuis le chemin spécifié
        model.load_model("/content/drive/MyDrive/stage/hh/Medical_Reglementation/models/biosentvec.crdownload")
except Exception as e:
      # Gestion des erreurs lors du chargement du modèle
      print(e)


# Define a Pydantic model to validate the input
class TextInput(BaseModel):
    text: str = "irbesartan, hydrochlorothiazide"
    countries: str = "Belgium - FPS Health-DGM"
    eudract: str = "2004-000020-32"
    disease: str = None 

def extract_regulation(drug, countries, eudract, disease):
    '''Extraction de la réglementation du médicament donné'''
    
    if drug in df["Substance active"].values:
    
      y = df["cluster_labels"][df["Substance active"] == drug].iloc[0]
      print(y)
      # Recherche de médicaments similaires dans le même cluster
      df_clus = df[df['cluster_labels'] == y]
    
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
        similar_medications_in_cluster = faiss_search_similar_medications(drug, disease, df_clus, 1)
        print(similar_medications_in_cluster)
        

    index_po, index_so, index_in, index_ex, index_ep = vector_index(df_clus)
    query_engine_po = index_po.as_query_engine()
    query_engine_so = index_so.as_query_engine()
    query_engine_in = index_in.as_query_engine()
    query_engine_ex = index_ex.as_query_engine()
    query_engine_ep = index_ep.as_query_engine()

    reponses = []
    response_po=query_engine_po.query(f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what is the Main objective of the trial?")
    reponses.append(response_po)
    response_so=query_engine_so.query(f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what are the Secondary objectives of the trial?")
    reponses.append(response_so)
    response_in=query_engine_in.query(f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what is the Principal inclusion criteria?")
    reponses.append(response_in)
    response_ex=query_engine_ex.query(f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what is the Principal exclusion criteria?")
    reponses.append(response_ex)
    response_ep=query_engine_ep.query(f"for EudraCT Number: {eudract}, 'Substance active': {drug} and Member State Concerned: {countries}, what is the Primary end point(s)?")
    reponses.append(response_ep)

    titles = ["Main objective of the trial", "Secondary objectives of the trial", "Principal inclusion criteria", "Principal exclusion criteria", "Primary end point(s)"]
    parts = [f"{title}\n\n{paragraph}" for title, paragraph in zip(titles, reponses)]
    

    return '\n\n'.join(parts)
    
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
@app.post("/apiv1/regulation/get-regulation")
async def get_regulation(drug: TextInput):
    # You can perform text processing here
    start_time = time.time()
    regulation_text = extract_regulation(drug.text, drug.countries, drug.eudract, drug.disease)
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
