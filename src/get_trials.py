import time
import uvicorn
from colabcode import ColabCode
from fastapi import FastAPI
from pydantic import BaseModel
from faiss_M import faiss_search_similar_medications
from llm import llm
from utils import process
import pickle
import os
import pandas as pd
import numpy as np
import sent2vec
from vector_store import vector_index
from llama_index.core.prompts.prompts import SimpleInputPrompt

app = FastAPI()
cc =  ColabCode(port=8002, code=False)
df = pd.read_csv("./data/processed/data_cluster.csv")

# Initialisation du modèle Sent2Vec
model = sent2vec.Sent2vecModel()
try:
      # Chargement du modèle depuis le chemin spécifié
        model.load_model("./models/biosentvec.crdownload")
except Exception as e:
      # Gestion des erreurs lors du chargement du modèle
      print(e)


# Define a Pydantic model to validate the input
class TextInput(BaseModel):
    # text: str = "Povidone, amidon prégélatinisé, carboxyméthylamidon sodique (type A), talc, stéarate de magnésium"
    text: str = "ritonavir"
    contries: str= "Spain - AEMPS"
    eudract: str = "2008-001611-38"



def extract_regulation(drug, contries, eudract):
    '''Extraction de la réglementation du médicament donné'''
    drug = process(drug)
    print(df[0:1])
    
    # Intégration du texte du médicament et d'une phrase représentative de la maladie
    embedded_drug = model.embed_sentence(drug)
    disease = df["Diseases"][df["Substance active"] == drug].iloc[0]
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
    

    index_po, index_so, index_in, index_ex, index_ep = vector_index(df_clus)
    query_engine_po = index_po.as_query_engine()
    query_engine_so = index_so.as_query_engine()
    query_engine_in = index_in.as_query_engine()
    query_engine_ex = index_ex.as_query_engine()
    query_engine_ep = index_ep.as_query_engine()

    response_po=query_engine_po.query("for EudraCT Number: 2004-000088-92, what is the Name of Sponsorl?")
    return response_po
    
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
    print(drug)
    regulation_text = extract_regulation(drug.text.lower())
    # stopping the timer
    stop_time = time.time()
    elapsed_time = stop_time - start_time

    # formatting the elapsed time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(regulation_text)
    # Return the processed text as a response
    return {"regulation": regulation_text}


if __name__ == '__main__':
    cc.run_app(app=app)