import locale
# Définition de la fonction de préférence d'encodage de la locale pour toujours retourner "UTF-8"
locale.getpreferredencoding = lambda: "UTF-8"

# Importation de la classe HuggingFaceHub depuis le module langchain
from llama_index.llms.huggingface import HuggingFaceLLM

# Définition de la fonction llm qui crée et retourne un modèle de langage mixte
def llm(system_prompt):
    # Création d'une instance de la classe HuggingFaceHub avec les paramètres spécifiés
    llm_model = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=256,
            model_name="TheBloke/Mixtral-8x7B-v0.1-GPTQ",
            tokenizer_name="TheBloke/Mixtral-8x7B-v0.1-GPTQ",  # Identifiant du dépôt du modèle
            generate_kwargs={"temperature": 0.0, "do_sample": False},
            system_prompt=system_prompt,
    )
    # Retour de l'instance du modèle de langage mixte créée
    return llm_model
