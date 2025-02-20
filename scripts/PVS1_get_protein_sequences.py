import os
import requests
import pandas as pd


def fetch_uniprot_sequence(uniprot_id):
    """
    Fetches the protein sequence from UniProt given a UniProt ID.
    Returns the sequence as a string.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    print(f"url: {url}")
    response = requests.get(url)

    if response.status_code == 200:
        fasta_data = response.text
        sequence = "".join(fasta_data.split("\n")[1:])  # Remove header line
        return sequence
    else:
        print(f"Failed to fetch sequence for {uniprot_id}")
        return None


def save_uniprot_sequences(df, output_dir):
    """
    Extracts unique gene-UniProt ID pairs from the dataframe,
    fetches protein sequences, and saves them as FASTA files.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Get unique gene-UniProt ID pairs
    unique_entries = df[["Gene", "UniProt_ID"]].drop_duplicates()

    for _, row in unique_entries.iterrows():
        gene = row["Gene"]
        uniprot_id = row["UniProt_ID"]

        if pd.notna(uniprot_id):  # Ensure UniProt ID is valid
            sequence = fetch_uniprot_sequence(uniprot_id)

            if sequence:
                fasta_filename = os.path.join(output_dir, f"{gene}_{uniprot_id}.fasta")

                with open(fasta_filename, "w") as f:
                    f.write(f">{gene}_{uniprot_id}\n{sequence}\n")

                print(f"Saved: {fasta_filename}")


if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")

    # Load dataset
    file_path = "./data/nonACMGPLP_pLoF_allinfo_AA_simplified_with_UniProt_IDs.xlsx"
    df = pd.read_excel(
        file_path, dtype={"#CHROM": str, "REF": str, "ALT": str, "Gene": str}
    )

    print(df.head())

    # Define output directory for FASTA files
    output_dir = "./data/protein_sequences"

    # Call function to fetch and save sequences
    save_uniprot_sequences(df, output_dir)
