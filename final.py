import pandas as pd
import joblib
import os
pd.set_option('display.float_format', '{:.6f}'.format)

def process_data(file_path, desired_columns):
    try:
        # Read the CSV file with initial columns including 'Name'
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

        # Ensure 'SMILES' is included in the columns and remove duplicate 'ID'
        final_columns = ['ID', 'SMILES'] + [col for col in desired_columns if col != 'ID']

        df = df[final_columns]

        # Drop rows with any non-numeric values in the other columns
        df.dropna(subset=[col for col in desired_columns if col != 'ID'], inplace=True)

        # Save to CSV
        df.to_csv(file_path, index=False, sep=';')
    except Exception as e:
        print(f"An error occurred: {e}")

# Columns to keep and their order
desired_columns = [
    "ID", "SHBd", "MLFER_S", "nBase", "maxsssN", "MLFER_BH", "minssNH", "MDEN-23", 
    "MLFER_BO", "MATS4e", "ATSC1c", "GATS4e", "MATS1v", "maxsCH3", "MATS4m", 
    "ATSC1v", "MDEO-11", "AATSC1v", "ETA_dEpsilon_D", "MLFER_L", "MATS1c", "SdO", 
    "WTPT-3", "ATSC4e", "AATSC4s", "GATS1c", "GATS1e", "WTPT-4", "WTPT-5", "C1SP3", 
    "MATS4s", "AATSC4e", "MDEN-22", "ATSC4s", "ATSC4m", "AATS2i", "MLFER_E", "SpDiam_Dt", 
    "SHAvin", "BCUTc-1h", "SaaN", "ETA_Beta_s", "GATS6p", "GATS6e", "SpMin1_Bhp", 
    "MATS3s", "AMR", "GATS3c", "JGI5", "GATS1s"
]

# Process the data
file_path = 'data/dataFLT3.csv'
process_data(file_path, desired_columns)

# Load the trained model
loaded_model = joblib.load('modelFLT3.pkl')

# Load new data
new_data = pd.read_csv(file_path, delimiter=';')

# Assuming the first column of the new dataset is "ID"
ids = new_data.iloc[:, 0]

# Assuming the rest of the columns are features
X_new = new_data.iloc[:, 2:]  # Adjusted to skip 'SMILES'

# Make predictions with the loaded model
y_pred = loaded_model.predict(X_new)

# Load similarity data
similarity_data = pd.read_csv('data/sp.csv', delimiter=';')

# Select the second column (index 1) for the similarity percentage
similarity_data['%Similarity'] = similarity_data.iloc[:, 1]

# Create a DataFrame with IDs, SMILES, predictions
results = pd.DataFrame({
    'ID': ids,
    'SMILES': new_data['SMILES'],  # Add the SMILES column
    'pIC50': y_pred
})

# Add a new column that is 10 raised to the power of pIC50
results['IC50 [nM]'] = results['pIC50'].apply(lambda x: (10 ** -x) * 1000000000)

def classify_activity(pIC50):
    if pIC50 <= -8:
        return 'High'
    elif -8 < pIC50 <= -6:
        return 'Medium'
    else:
        return 'Low'

# Add a new column
results['Relative activity'] = results['pIC50'].apply(classify_activity)

# Merge the similarity data with the 'results' DataFrame based on 'ID'
results = pd.merge(results, similarity_data[['ID', '%Similarity']], on='ID')

# Function to categorize the values in '%Similarity'
def categorize_similarity(value):
    if value >= 80:
        return 'High'
    elif 65 <= value < 80:
        return 'Medium'
    else:
        return 'Low'

# Apply the categorization function to the '%Similarity' column
results['Reliability'] = results['%Similarity'].apply(categorize_similarity)

# Remove the '%Similarity' column as it's no longer needed
results.drop(columns=['%Similarity'], inplace=True)

# Sort the DataFrame by 'IC50 [nM]' in ascending order
results.sort_values(by='IC50 [nM]', ascending=True, inplace=True)

def print_centered_df(df, columns):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Determine the maximum width for each column
    max_widths = {}
    for column in columns:
        if df_copy[column].dtype in [float, 'float64']:  # Check if the column is of float type
            # Format with 6 decimals using .loc
            df_copy.loc[:, column] = df_copy[column].map('{:.6f}'.format)
        max_widths[column] = max(len(str(column)), df_copy[column].astype(str).str.len().max())

    # Format and center each value of each column
    formatted_data = {column: df_copy[column].astype(str).str.center(max_widths[column]) for column in columns}
    formatted_df = pd.DataFrame(formatted_data)

    # Center the headers
    formatted_columns = [str(column).center(max_widths[column]) for column in formatted_df.columns]

    # Print the result
    print(" ".join(formatted_columns))
    print(formatted_df.to_string(index=False, header=False))


# Print the results, excluding the 'SMILES' column
print(" ")
print("*" * 53)                         
print("*" + " " * 51 + "*")             
print("*                     RESULTS:" + " " * 22 + "*") 
print("*" + " " * 51 + "*")  
print("*" * 53)

columns_to_print = ['ID', 'pIC50', 'IC50 [nM]', 'Relative activity', 'Reliability']
if len(results) > 5:
    print_centered_df(results[columns_to_print].head(), columns_to_print)
    print("... For more results, please export.")
else:
    print_centered_df(results[columns_to_print], columns_to_print)

print("*" * 53)
print(" ")


# List all files in the directory
files_in_directory = os.listdir(os.getcwd())

# Filter and delete .smi files
for file in files_in_directory:
    if file.endswith('.smi'):
        os.remove(os.path.join(os.getcwd(), file))

# Ask the user if they want to export the results
while True:
    export_answer = input("Do you want to export the results to a CSV file? (yes/no): ")
    if export_answer.lower() in ['yes', 'no']:
        break
    else:
        print("Please enter 'yes' or 'no'.")

if export_answer.lower() == 'yes':
    # Export DataFrame to a CSV file including the similarity percentage
    results.to_csv('Report.csv', sep=';', index=False)
    print("Results have been exported as Report.csv.")