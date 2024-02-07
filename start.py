import csv

def save_smiles(smiles, filename):
    with open(filename, 'w') as file:
        file.write(smiles)

def process_single_smiles():
    smiles = input("Enter the SMILES code: ")
    save_smiles(smiles, '1.smi')

def process_smiles_list():
    csv_filename = input("Enter the name of the CSV file (including .csv extension): ")
    try:
        with open(csv_filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                for smiles in row:
                    save_smiles(smiles, f'{i+1}.smi')
    except FileNotFoundError:
        print("CSV file not found.")

# Script execution starts here
while True:
    print("\nChoose an option:")
    print("------------------")
    print("1) Single SMILES")
    print("2) List of SMILES from a CSV file")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        process_single_smiles()
        break  
    elif choice == '2':
        process_smiles_list()
        break  
    else:
        print("Invalid choice. Please enter 1 or 2.")


