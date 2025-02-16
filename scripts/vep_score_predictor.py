import os
import argparse
import pandas as pd
import torch
import json

# from Bio import SeqIO

# from transformers import (
#     AutoConfig,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     EsmForMaskedLM,  # only used for ESM model
# )
import requests


# File path for caching protein sequences
CACHE_FILE = "./data/gene_protein_cache.json"
# Define cache file for gene-to-Ensembl ID mappings
GENE_ID_CACHE_FILE = "./data/gene_id_cache.json"


def load_variant_file(filename, sheet_name=0):
    """
    Loads an Excel file containing genetic variant data.

    Expected Columns:
      #CHROM, POS, REF, ALT, Gene, variant, num_patients, VarType,
      InterVar_automated, CLNSIG, ExonicFunc_refGene, Func_refGene, ExonicFunc_knownGene

    Returns:
      - A Pandas DataFrame with properly formatted data.
    """
    # Load the Excel file (detects sheet automatically)
    df = pd.read_excel(
        filename,
        sheet_name=sheet_name,
        dtype={"#CHROM": str, "REF": str, "ALT": str, "Gene": str, "variant": str},
    )

    # Convert numeric columns
    df["POS"] = df["POS"].astype(int)  # Genomic position
    df["num_patients"] = df["num_patients"].astype(int)  # Number of patients

    return df


def load_cached_protein_sequences():
    """Load cached protein sequences from JSON file if it exists."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cached_protein_sequences(cache):
    """Save updated protein sequence cache to JSON file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def get_protein_sequence(gene_name, cache):
    """
    Fetch the protein sequence for a given gene from Ensembl.
    If sequence exists in cache, return from cache.
    """
    if gene_name in cache:
        return cache[gene_name]

    url = f"https://rest.ensembl.org/sequence/id/{gene_name}?type=protein"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        protein_seq = response.text.strip()
        cache[gene_name] = protein_seq  # Cache the sequence
        save_cached_protein_sequences(cache)  # Update cache file
        return protein_seq
    else:
        return None  # If no sequence is found


def load_cached_gene_ids():
    """Load cached gene-to-Ensembl ID mappings from a JSON file."""
    if os.path.exists(GENE_ID_CACHE_FILE):
        with open(GENE_ID_CACHE_FILE, "r") as f:
            try:
                return json.load(f)  # Load JSON file safely
            except json.JSONDecodeError:
                return {}  # Return empty if file is corrupted
    return {}


def save_cached_gene_ids(cache):
    """Save updated gene-to-Ensembl ID mappings in a JSON file."""
    with open(GENE_ID_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def get_ensembl_gene_id(gene_name, cache):
    """
    Fetches the Ensembl Gene ID for a given gene name.
    Uses cache to minimize API calls.
    """
    if gene_name in cache:
        return cache[gene_name]  # Return cached ID

    url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_name}?content-type=application/json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        ensembl_id = data.get("id")  # Get Ensembl Gene ID
        if ensembl_id:
            print(f"Gene:{gene_name}-ID: {ensembl_id}")
            cache[gene_name] = ensembl_id  # Store gene name as key
            save_cached_gene_ids(cache)  # Update cache file
            return ensembl_id
    return None  # If not found


def get_protein_variant(chrom, pos, ref, alt):
    """
    Queries Ensembl VEP to translate a genomic variant into a protein-level variant.
    """
    url = "https://rest.ensembl.org/vep/human/region/"
    headers = {"Content-Type": "application/json"}

    data = {"variants": [f"{chrom} {pos} {ref} {alt}"]}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        results = response.json()
        if results and "transcript_consequences" in results[0]:
            for transcript in results[0]["transcript_consequences"]:
                if "protein_start" in transcript and "amino_acids" in transcript:
                    protein_change = transcript.get(
                        "hgvsp", "Unknown"
                    )  # e.g., p.Ala120Thr
                    print(f"Protein Change: {protein_change}")
                    return protein_change
    return None  # If no protein-level annotation is available


# Convert GRCh37 to GRCh38
def convert_grch37_to_grch38(chrom, pos):
    """
    Converts GRCh37 coordinates to GRCh38 using Ensembl's assembly converter API.
    """
    url = f"https://rest.ensembl.org/map/human/GRCh37/{chrom}:{pos}..{pos}/GRCh38?"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if "mappings" in data and len(data["mappings"]) > 0:
            return data["mappings"][0]["mapped"]["start"]  # Return converted position
    return None  # If conversion fails


# Query VEP API to retrieve transcript ID
def get_transcript_id(chrom, pos, ref, alt):
    """
    Queries Ensembl VEP API to retrieve the transcript ID (ENST) for a given genomic variant.
    Converts GRCh37 to GRCh38 before querying if necessary.
    """
    # Convert position to GRCh38
    new_pos = convert_grch37_to_grch38(chrom, pos)

    if new_pos is None:
        print(f"Failed to convert position: {chrom}:{pos}")
        return None  # Skip if conversion fails

    # Query VEP API using GRCh38 position
    url = f"https://rest.ensembl.org/vep/human/region/{chrom}:{new_pos}/{ref}/{alt}?"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        results = response.json()
        if results and "transcript_consequences" in results[0]:
            for transcript in results[0]["transcript_consequences"]:
                if "transcript_id" in transcript:
                    return transcript["transcript_id"]  # Return first found ENST ID

    return None  # If no transcript is found


# def get_protein_sequence(gene_name):
#     """
#     Fetches the wild-type protein sequence for a given gene from Ensembl.
#     """
#     url = f"https://rest.ensembl.org/sequence/id/{gene_name}?type=protein"
#     headers = {"Content-Type": "application/json"}

#     response = requests.get(url, headers=headers)

#     if response.status_code == 200:
#         return response.text.strip()  # Returns the protein sequence
#     else:
#         return None  # If no sequence is found, return None


def get_vep_score(df, model, tokenizer, device, wt_seq, mut_seq):
    # Process each variant
    vep_scores = []
    for _, row in df.iterrows():
        gene = row["Gene"]
        pos = row["POS"]
        ref = row["REF"]
        alt = row["ALT"]

        # Fetch the protein sequence
        wt_seq = get_protein_sequence(gene)


#
def read_args():
    parser = argparse.ArgumentParser(description="Predict Protein Mutation Using PLM.")

    parser.add_argument(
        "--loadFlag",
        action="store_true",  # turn it into a boolean flag
        default=False,
        help="If set, load the model from local pickle directory (default: False)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        choices=["togethercomputer/evo-1-131k-base", "facebook/esm2_t6_8M_UR50D"],
        help="Name of the PLM to use (default: facebook/esm2_t6_8M_UR50D)",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="./data/BRCA1_rare_variants_ESM2_predictions_small.csv",
        help="Output filename (default: ./data/ESM2_predictions_small.csv)",
    )

    return parser.parse_args()


def load_model(model_name, load_flag=True):
    """
    Loads the specified model and tokenizer.
    If load_flag is False, loads from Hugging Face Hub (online).
    If load_flag is True, loads from local 'pickle' directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "togethercomputer/evo-1-131k-base":
        # ========= EVO: Causal LM =========
        if not load_flag:
            # 1) Load from Hugging Face
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True, revision="1.1_fix"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, config=config, trust_remote_code=True, revision="1.1_fix"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, revision="1.1_fix"
            )
            model.to(device)
            model.eval()
            return model, tokenizer, device
        else:
            # 2) Load from local pickle
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            pickle_dir = os.path.join(base_dir, "pickle")
            model_path = os.path.join(
                pickle_dir, model_name.replace("/", "_") + "_model"
            )
            tokenizer_path = os.path.join(
                pickle_dir, model_name.replace("/", "_") + "_tokenizer"
            )

            print(f"Loading EVO model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.to(device)
            model.eval()

            print(f"Loading EVO tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            return model, tokenizer, device

    elif model_name == "facebook/esm2_t6_8M_UR50D":
        # ========= ESM: Masked LM =========
        if not load_flag:
            # 1) Load from Hugging Face
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = EsmForMaskedLM.from_pretrained(model_name).to(device)
            model.eval()
            return model, tokenizer, device
        else:
            # 2) Load from local pickle
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            pickle_dir = os.path.join(base_dir, "pickle")
            model_path = os.path.join(
                pickle_dir, model_name.replace("/", "_") + "_model"
            )
            tokenizer_path = os.path.join(
                pickle_dir, model_name.replace("/", "_") + "_tokenizer"
            )

            print(f"Loading ESM model from {model_path}")
            model = EsmForMaskedLM.from_pretrained(model_path)
            model.to(device)
            model.eval()

            print(f"Loading ESM tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            return model, tokenizer, device

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":
    print(f"Curr.dir: {os.getcwd()}")
    # Example usage
    file_path = "./data/nonACMGPLP_pLoF.xlsx"  # Update with actual file path
    df = load_variant_file(file_path)

    df["ENST_ID"] = df.apply(
        lambda row: get_transcript_id(
            row["#CHROM"], row["POS"], row["REF"], row["ALT"]
        ),
        axis=1,
    )

    # df = df[df["Gene"].notnull()]  # Remove NaN values in Gene column
    # unique_genes = df["Gene"].unique().tolist()

    # print(f"Number of genes:{len(unique_genes)}")

    # # Add a column for protein-level variants
    # df["Protein_Variant"] = df.apply(
    #     lambda row: get_protein_variant(
    #         row["#CHROM"], row["POS"], row["REF"], row["ALT"]
    #     ),
    #     axis=1,
    # )

    # Display first few rows
    print(df.head())
    # Save updated dataset
    output_file = "annotated_transcripts.xlsx"
    df.to_excel(output_file, index=False)

    # Load existing cache
    # gene_id_cache = load_cached_gene_ids()
    # gene_cache = load_cached_protein_sequences()

    # # Process each gene
    # gene_mappings = {}  # Store gene-to-Ensembl ID mappings

    # # Fetch and cache sequences
    # for gene in unique_genes:
    #     print(f"Fetching sequence for: {gene}")
    #     get_ensembl_gene_id(gene, gene_id_cache)
    #     # sequence = get_protein_sequence(gene, gene_cache)
    #     # print(
    #     #     f"Gene: {gene}, Sequence Length: {len(sequence) if sequence else 'Not Found'}"
    #     # )

    # # Save the updated cache after fetching
    # save_cached_protein_sequences(gene_cache)

    # print(f"Total Cached Genes: {len(gene_cache)}")

    # # 1) Read command-line arguments
    # args = read_args()
    # print(f"LoadFlag: {args.loadFlag}")
    # print(f"Model: {args.model}")
    # print(f"Output filename: {args.output_filename}")
