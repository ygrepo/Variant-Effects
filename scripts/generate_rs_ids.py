import os
import json
import pandas as pd
import requests

# Define cache file for HGNC mappings
CACHE_FILE = "./data/nonACMGPLP_pLoF_transcript_hgnc_cache.json"


def load_cache():
    """Load cached transcript mappings from JSON file if it exists."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                return json.load(f)  # Load JSON safely
            except json.JSONDecodeError:
                return {}  # Return empty cache if file is corrupted
    return {}


def save_cache(cache):
    """Save updated transcript mappings to JSON file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


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


def get_hgnc_from_transcript(transcript_id, chrom, pos, ref, alt, gene_symbol, cache):
    """
    Queries Ensembl VEP API to retrieve HGNC ID for a given transcript mutation.
    Ensures the retrieved `HGNC ID` matches the provided `gene_symbol`.
    Uses caching to avoid redundant queries.
    """
    if pd.isna(transcript_id) or transcript_id is None:
        return None  # Skip if transcript_id is missing

    if pd.isna(chrom) or pd.isna(pos) or pd.isna(ref) or pd.isna(alt):
        return None  # Skip if any genomic data is missing

    # Convert GRCh37 position to GRCh38 if necessary
    new_pos = convert_grch37_to_grch38(chrom, pos)
    if new_pos is None:
        print(f"Failed to convert position: {chrom}:{pos}")
        return None  # Skip if conversion fails

    api_key = f"{transcript_id}:{chrom}:{new_pos}:{ref}:{alt}"
    print(f"Querying hgnc: {api_key}")

    # Check cache before querying
    if api_key in cache:
        print(f"Cache hit for {api_key}")
        return cache[api_key]

    # Query VEP API using GRCh38 position
    url = f"https://rest.ensembl.org/vep/human/region/{chrom}:{new_pos}/{ref}/{alt}?"
    print(f"url:{url}")
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        results = response.json()
        best_hgnc = None

        if results and "transcript_consequences" in results[0]:
            for transcript in results[0]["transcript_consequences"]:
                # Only consider protein-coding genes and ensure gene symbol matches
                if transcript.get("biotype") == "protein_coding":
                    hgnc_id = transcript.get("hgnc_id")
                    transcript_gene_symbol = transcript.get("gene_symbol")
                    print(
                        f"Gene: {gene_symbol} - Gene Returned: {transcript_gene_symbol} - HGNC ID: {hgnc_id}"
                    )

                    if hgnc_id and transcript_gene_symbol == gene_symbol:
                        print(f"Found matching HGNC ID: {hgnc_id} for {gene_symbol}")
                        best_hgnc = hgnc_id  # Store the HGNC ID
                        break  # Prioritize first found protein-coding transcript for matching gene

        if best_hgnc:
            cache[api_key] = best_hgnc
            save_cache(cache)
            return best_hgnc

    print(f"No matching HGNC ID found for {gene_symbol} at {chrom}:{new_pos}")
    cache[api_key] = None  # Store failed queries to avoid retrying
    save_cache(cache)
    return None  # No HGNC found


if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")

    # Load dataset
    file_path = "./data/nonACMGPLP_pLoF_annotated_transcripts.xlsx"
    df = pd.read_excel(
        file_path,
        dtype={
            "#CHROM": str,
            "POS": int,
            "REF": str,
            "ALT": str,
            "ENST_ID": str,
            "Gene": str,  # Ensure the gene name is loaded for matching
        },
    )

    # Load existing cache
    cache = load_cache()

    # Step 1: Convert `ENST` (Transcript) to `HGNC ID` with gene matching
    if "ENST_ID" in df.columns and "Gene" in df.columns:
        df["HGNC_ID"] = df.apply(
            lambda row: pd.Series(
                get_hgnc_from_transcript(
                    row["ENST_ID"],
                    row["#CHROM"],
                    row["POS"],
                    row["REF"],
                    row["ALT"],
                    row["Gene"],
                    cache,
                )
                if pd.notna(row["ENST_ID"])
                else None
            ),
            axis=1,
        )

    # Save updated dataset
    output_file = "./data/nonACMGPLP_pLoF_annotated_hgnc.xlsx"
    df.to_excel(output_file, index=False)

    print(f"Updated dataset saved to {output_file}!")
