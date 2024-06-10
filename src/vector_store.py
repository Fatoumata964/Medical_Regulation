from llama_index.core import VectorStoreIndex
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.service_context import ServiceContext
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.core import Document
from llm import llm

mixtral = llm()

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))


def vector_index(df_final):
    p_objectives = [
        Document(
            text= f"Main objective of the trial: " + str(row['E.2.1 Main objective of the trial']), 
            metadata={
                'EudraCT Number': row['A.2 EudraCT number'],
                'Member State Concerned:': row['A.1 Member State Concerned'],
                'cluster': row['cluster_labels'],
                'Substance active': row['Substance active']
            }
        )
        for _, row in df_final.iterrows()
    ]
    s_objectives = [
        Document(
            text= f"Secondary objectives of the trial: " + str(row['E.2.2 Secondary objectives of the trial']), 
            metadata={
                'EudraCT Number': row['A.2 EudraCT number'],
                'Member State Concerned:': row['A.1 Member State Concerned'],
                'cluster': row['cluster_labels'],
                'Substance active': row['Substance active']
            }
        )
        for _, row in df_final.iterrows()
    ]
    inclusion = [
        Document(
            text= f"Principal inclusion criteria: " + str(row['E.3 Principal inclusion criteria']), 
            metadata={
                'EudraCT Number': row['A.2 EudraCT number'],
                'Member State Concerned:': row['A.1 Member State Concerned'],
                'cluster': row['cluster_labels'],
                'Substance active': row['Substance active']
            }
        )
        for _, row in df_final.iterrows()
    ]
    exclusion = [
        Document(
            text= f"Principal exclusion criteria: " + str(row['E.4 Principal exclusion criteria']), 
            metadata={
                'EudraCT Number': row['A.2 EudraCT number'],
                'Member State Concerned:': row['A.1 Member State Concerned'],
                'cluster': row['cluster_labels'],
                'Substance active': row['Substance active']
            }
        )
        for _, row in df_final.iterrows()
    ]
    endpoint = [
        Document(
            text= f"Primary end point: " + str(row['E.5.1 Primary end point']), 
            metadata={
                'EudraCT Number': row['A.2 EudraCT number'],
                'Member State Concerned:': row['A.1 Member State Concerned'],
                'cluster': row['cluster_labels'],
                'Substance active': row['Substance active']
            }
        )
        for _, row in df_final.iterrows()
    ]

    service_context=ServiceContext.from_defaults(
        chunk_size=1024,
        llm=mixtral,
        embed_model=embed_model
    )

    index_po=VectorStoreIndex.from_documents(p_objectives,service_context=service_context)
    index_so=VectorStoreIndex.from_documents(s_objectives,service_context=service_context)
    index_in=VectorStoreIndex.from_documents(inclusion,service_context=service_context)
    index_ex=VectorStoreIndex.from_documents(exclusion,service_context=service_context)
    index_ep=VectorStoreIndex.from_documents(endpoint,service_context=service_context)
    
    return index_po, index_so, index_in, index_ex, index_ep
