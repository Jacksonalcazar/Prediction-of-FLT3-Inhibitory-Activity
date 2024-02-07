import pandas as pd
import joblib
import os  

def process_data(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Replace 'AUTOGEN_' in 'Name' column and convert it to numeric
        df['Name'] = df['Name'].str.replace('AUTOGEN_', '')
        df['Name'] = pd.to_numeric(df['Name'], errors='coerce')

        # Rename 'Name' column to 'ID'
        df.rename(columns={'Name': 'ID'}, inplace=True)

        # Sort by 'ID'
        df.sort_values(by='ID', inplace=True)

        # Read SMILES from .smi files and add them to a new column
        smiles_column = []
        for id in df['ID']:
            smi_file = f"{int(id)}.smi"
            if os.path.exists(smi_file):
                with open(smi_file, 'r') as file:
                    smiles = file.read().strip()
                    smiles_column.append(smiles)
            else:
                smiles_column.append('')  # In case the .smi file does not exist

        df.insert(1, 'SMILES', smiles_column)  # Insert the 'SMILES' column after 'ID'

        # Drop rows with any non-numeric values in the other columns, except 'SMILES'
        numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        df.dropna(subset=numeric_cols, inplace=True)

        # Save to CSV
        df.to_csv(file_path, index=False, sep=';')
    except Exception as e:
        print(f"An error occurred: {e}")

# Process the data
file_path = 'data/Fingerprints.csv'
process_data(file_path)

def calculate_tanimoto_similarity(fp1, fp2):
    intersection = sum(a and b for a, b in zip(fp1, fp2))
    union = sum(a or b for a, b in zip(fp1, fp2))
    similarity = intersection / union if union != 0 else 0
    return round(similarity * 100, 1)

file_db = 'data/FP_train.csv'

df_list = pd.read_csv(file_path, delimiter=';', index_col='ID')
df_db = pd.read_csv(file_db, delimiter=';', index_col='ID')

# Calculating Tanimoto similarity
results = []
for id_list, fp_list in df_list.iterrows():
    max_similarity = 0
    for id_db, fp_db in df_db.iterrows():
        similarity = calculate_tanimoto_similarity(fp_list[2:], fp_db[2:])
        max_similarity = max(max_similarity, similarity)
    results.append({'ID': id_list, '%Similarity': max_similarity})

# Create DataFrame of results and export to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('data/sp.csv', sep=';', index=False)


