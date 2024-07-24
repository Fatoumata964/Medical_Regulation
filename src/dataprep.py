import os
import re
import pandas as pd
import ast

def process_files(directory, data_cluster_path):
    # Initialisation d'une liste pour stocker les données extraites
    text_data = []

    # Critère de filtre pour les colonnes utilisant une expression régulière
    pattern = re.compile(r'^[A-Z]\..*: ')
    stop_pattern = re.compile(r'^[A-Z]\.\d+ ')

    # Lecture des fichiers texte avec filtrage
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                file_data = {}
                current_key = None
                current_value = []

                for line in file:
                    if pattern.match(line):
                        if current_key:
                            # Sauvegarder la clé précédente et ses valeurs
                            value_to_add = ' '.join(current_value).strip()
                            if current_key in file_data:
                                if isinstance(file_data[current_key], list):
                                    # Ajouter uniquement des valeurs non dupliquées
                                    if value_to_add not in file_data[current_key]:
                                        file_data[current_key].append(value_to_add)
                                else:
                                    file_data[current_key] = [file_data[current_key], value_to_add]
                            else:
                                file_data[current_key] = [value_to_add]

                        # Nouvelle clé trouvée
                        parts = line.strip().split(': ', 1)
                        if len(parts) == 2:
                            current_key, value = parts
                            current_value = [value.strip()]
                    else:
                        # Vérifie si la ligne correspond au stop_pattern
                        if stop_pattern.match(line):
                            if current_key:
                                value_to_add = ' '.join(current_value).strip()
                                if current_key in file_data:
                                    if isinstance(file_data[current_key], list):
                                        if value_to_add not in file_data[current_key]:
                                            file_data[current_key].append(value_to_add)
                                    else:
                                        file_data[current_key] = [file_data[current_key], value_to_add]
                                else:
                                    file_data[current_key] = [value_to_add]
                            current_key = None
                            current_value = []
                        else:
                            # Ligne de continuation
                            current_value.append(line.strip())

                # Sauvegarder la dernière clé et ses valeurs
                if current_key:
                    value_to_add = ' '.join(current_value).strip()
                    if current_key in file_data:
                        if isinstance(file_data[current_key], list):
                            if value_to_add not in file_data[current_key]:
                                file_data[current_key].append(value_to_add)
                        else:
                            file_data[current_key] = [file_data[current_key], value_to_add]
                    else:
                        file_data[current_key] = [value_to_add]

                text_data.append(file_data)

    # Normaliser les données pour la création du DataFrame
    expanded_data = []

    for data in text_data:
        # Extraire les produits
        product_names = data.get('D.3.1 Product name', []) + data.get('D.2.1.1.1 Trade name', [])

        # Si aucun produit n'est trouvé, ajouter une entrée avec les autres données
        if not product_names:
            expanded_data.append(data)
        else:
            for product in product_names:
                new_entry = data.copy()
                new_entry['D.3.1 Product name'] = product
                # Supprimer la colonne 'D.2.1.1.1 Trade name' si elle existe
                new_entry.pop('D.2.1.1.1 Trade name', None)
                expanded_data.append(new_entry)

    # Création du DataFrame
    df = pd.DataFrame(expanded_data)

    def clean_column_name(name):
        cleaned_name = re.sub(r'\s*\(.*?\)\s*', '', name)
        cleaned_name = cleaned_name.strip()
        return cleaned_name

    df.columns = [clean_column_name(col) for col in df.columns]

    df = df.groupby(axis=1, level=0).first()
    threshold = 100
    cols_to_drop = df.columns[df.isnull().sum() > threshold]
    df.drop(cols_to_drop, axis=1, inplace=True)

    df2 = pd.read_csv(data_cluster_path)
    ndf = df2[['cluster_labels', 'Nom du médicament', 'Substance active', 'Diseases']]
    df_final = pd.merge(ndf, df, left_on='Nom du médicament', right_on='D.3.1 Product name', how='inner')

    # Appliquer les transformations à toutes les colonnes
    for col in df_final.columns:
        df_final[col].fillna('', inplace=True)
        df_final[col] = df_final[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Fonction pour convertir les valeurs en 'Yes' ou 'No' selon le cas
    def convert_to_yes_no(value):
        if isinstance(value, str) and value.startswith("Yes"):
            return 'Yes'
        elif isinstance(value, str) and value.startswith("No"):
            return 'No'
        return value

    # Appliquer la fonction à toutes les colonnes
    for col in df_final.columns:
        df_final[col] = df_final[col].apply(convert_to_yes_no)

    # Générer la colonne 'combined_key'
    df_final['combined_key'] = df_final[['E.2.1 Main objective of the trial', 'A.2 EudraCT number', 'A.1 Member State Concerned', 'Substance active']].astype(str).agg('-'.join, axis=1)

    # Supprimer les duplicatas
    df_final = df_final.drop_duplicates(subset=['combined_key'])

    return df_final

if __name__ == "__main__":
    directory = './data/protocol'
    data_cluster_path = './data/data_cluster.csv'
    final_df = process_files(directory, data_cluster_path)
    final_df.to_csv('final_output.csv', index=False)
