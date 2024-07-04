import requests
import pandas as pd
import os
import time

def request_with_retry(url, max_retries=5, delay=5):
    """Attempt to make an HTTP request with a defined number of retries."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response
        except requests.exceptions.RequestException as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(delay)
    return None  

def fetch_cids_by_smiles(smiles, threshold):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{smiles}/cids/JSON?Threshold={threshold}'
    response = request_with_retry(url)
    if response:
        try:
            return response.json()['IdentifierList']['CID']
        except KeyError:
            return []
    else:
        return []

def fetch_smiles_by_cid(cid):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON'
    response = request_with_retry(url)
    if response:
        return response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
    else:
        return None

try:
    df = pd.read_csv('SMILES.csv', delimiter=";")
    smiles_of_interest = df['SMILES']
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Prompt the user to enter the similarity threshold
threshold = input("Enter the structural similarity percentage to search for: ")

temp_files = []

for i, smiles in enumerate(smiles_of_interest):
    cids = fetch_cids_by_smiles(smiles, threshold)
    cids_smiles = [{'CID': cid, 'SMILES': fetch_smiles_by_cid(cid)} for cid in cids if fetch_smiles_by_cid(cid)]

    if cids_smiles:
        temp_df = pd.DataFrame(cids_smiles)
        temp_filename = f'temp_result_{i}.csv'
        temp_df.to_csv(temp_filename, sep=';', index=False)
        temp_files.append(temp_filename)

# Concatenate all temporary files into a single file
if temp_files:
    combined_df = pd.concat([pd.read_csv(f, delimiter=';') for f in temp_files])
    combined_df.to_csv('CID_SMILES_NEW.csv', sep=';', index=False)

    # Delete the temporary files
    for f in temp_files:
        os.remove(f)
else:
    print("No results found.")
