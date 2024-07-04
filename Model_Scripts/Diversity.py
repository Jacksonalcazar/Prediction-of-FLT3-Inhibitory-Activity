import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.ML.Cluster import Butina
import matplotlib.pyplot as plt

# Function to check if a SMILES string is valid
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Load the CSV file
file_path = 'Data.csv'
data = pd.read_csv(file_path, delimiter=';')

# Get the SMILES strings and filter out invalid values
smiles_list = data['SMILES'].dropna().astype(str)
valid_smiles_list = [smiles for smiles in smiles_list if is_valid_smiles(smiles)]

# Convert valid SMILES strings to RDKit molecules
molecules = [Chem.MolFromSmiles(smiles) for smiles in valid_smiles_list]

# Calculate molecular fingerprints (MACCS keys)
fingerprints = [MACCSkeys.GenMACCSKeys(mol) for mol in molecules]

# Calculate the distance list
num_molecules = len(fingerprints)
distances = []
for i in range(1, num_molecules):
    for j in range(i):
        distance = 1 - TanimotoSimilarity(fingerprints[i], fingerprints[j])
        distances.append(distance)

# Use the Butina clustering algorithm
threshold = 0.3  # This is a typical threshold for diversity
clusters = Butina.ClusterData(distances, num_molecules, threshold, isDistData=True)

# Create a bar plot showing the size of each cluster
cluster_sizes = [len(cluster) for cluster in clusters]
cluster_ids = range(len(cluster_sizes))

plt.figure(figsize=(6, 7))
plt.bar(cluster_ids, cluster_sizes, color='blue')
plt.xlabel('Cluster ID')
plt.ylabel('Cluster Size')
plt.show()

# Calculate the percentage of molecules in each cluster
total_molecules = sum(cluster_sizes)
percentages = [(size / total_molecules) * 100 for size in cluster_sizes]

# Export the x (Cluster ID), y (Cluster Size), and percentages to a CSV file with ';' delimiter
cluster_data = pd.DataFrame({
    'Cluster ID': cluster_ids,
    'Cluster Size': cluster_sizes,
    'Percentage': percentages
})
cluster_data.to_csv('cluster_results.csv', sep=';', index=False)

