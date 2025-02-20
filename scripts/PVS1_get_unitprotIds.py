import os
import requests
import pandas as pd


def get_uniprot_id(gene_name, taxon_id="9606"):
    """
    Fetches UniProt ID for a given gene name in the specified organism (default: human, taxon ID 9606).
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search?"
    query = f"query=gene:{gene_name}+AND+organism_id:{taxon_id}&fields=accession"
    url = base_url + query
    print(f"url: {url}")
    response = requests.get(url, headers={"Accept": "application/json"})

    if response.status_code == 200:
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]["primaryAccession"]  # Returns first UniProt ID
    return None  # Return None if not found


if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    # Load dataset
    file_path = "./data/nonACMGPLP_pLoF_allinfo_AA_simplified.xlsx"
    df = pd.read_excel(
        file_path, dtype={"#CHROM": str, "REF": str, "ALT": str, "Gene": str}
    )
    print(df.head())
    unique_genes = df["Gene"].unique()  # Get unique genes from the dataset

    # Fetch UniProt IDs for all genes
    gene_uniprot_mapping = {gene: get_uniprot_id(gene) for gene in unique_genes}

    # Add UniProt IDs to the dataframe
    df["UniProt_ID"] = df["Gene"].map(gene_uniprot_mapping)
    print(df.head())

    # Save the updated dataframe
    output_file = "./data/nonACMGPLP_pLoF_allinfo_AA_simplified_with_UniProt_IDs.xlsx"
    df.to_excel(output_file, index=False)
    print(f"UniProt IDs saved to {output_file}!")
