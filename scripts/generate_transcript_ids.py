import os
import json
import pandas as pd
import requests

# Define cache file for transcript results
CACHE_FILE = "./data/nonACMGPLP_pLoF_transcript_cache.json"


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


def get_best_transcript_id(chrom, pos, ref, alt, gene_symbol, cache):
    """
    Retrieves the best transcript ID (ENST) for a given genomic variant.
    Ensures the retrieved transcript matches the expected gene symbol.
    Uses cache to avoid redundant API calls.
    """
    if pd.isna(gene_symbol) or gene_symbol is None:
        return None  # Skip if no gene name is provided

    variant_key = f"{chrom}:{pos}:{ref}:{alt}"  # Unique identifier for caching
    print(
        f"Querying transcript for variant: {variant_key}, Expected Gene: {gene_symbol}"
    )

    # Check cache before querying Ensembl
    if variant_key in cache:
        print(f"Cache hit for variant: {variant_key}")
        return cache[variant_key]

    # Convert position to GRCh38
    new_pos = convert_grch37_to_grch38(chrom, pos)
    if new_pos is None:
        print(f"Failed to convert position: {chrom}:{pos}")
        return None  # Skip if conversion fails

    # Query VEP API using GRCh38 position
    url = f"https://rest.ensembl.org/vep/human/region/{chrom}:{new_pos}/{ref}/{alt}?"
    print(f"url: {url}")
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    # Normalize expected gene name for comparison
    expected_gene_symbol = gene_symbol.upper()

    if response.status_code == 200:
        results = response.json()
        best_transcript = None
        found_matching_gene = False

        if results and "transcript_consequences" in results[0]:
            for transcript in results[0]["transcript_consequences"]:
                enst_id = transcript.get("transcript_id")
                transcript_gene_symbol = transcript.get(
                    "gene_symbol", ""
                ).upper()  # Normalize case
                biotype = transcript.get("biotype", "")
                is_canonical = transcript.get("is_canonical", False)

                print(
                    f"Checking transcript {enst_id} - Gene Returned: {transcript_gene_symbol} - Expected: {expected_gene_symbol}"
                )

                # Ensure the transcript belongs to the correct gene
                if (
                    transcript_gene_symbol == expected_gene_symbol
                    and biotype == "protein_coding"
                ):
                    found_matching_gene = True
                    if is_canonical:
                        print(
                            f"✅ Found canonical transcript: {enst_id} for gene {expected_gene_symbol}"
                        )
                        cache[variant_key] = enst_id  # Cache canonical transcript
                        save_cache(cache)  # Save updated cache
                        return enst_id
                    best_transcript = enst_id  # Store as backup if non-canonical

        # Cache and return best available transcript if no canonical found
        if best_transcript:
            print(
                f"✅ Found best available transcript: {best_transcript} for gene {expected_gene_symbol}"
            )
            cache[variant_key] = best_transcript
            save_cache(cache)
            return best_transcript

        if not found_matching_gene:
            print(
                f"❌ No matching transcript found for {expected_gene_symbol} at {chrom}:{pos}"
            )

    return None  # If no transcript is found


if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")

    # Load dataset
    file_path = "./data/nonACMGPLP_pLoF.xlsx"
    df = pd.read_excel(
        file_path, dtype={"#CHROM": str, "REF": str, "ALT": str, "Gene": str}
    )

    # Load existing cache
    transcript_cache = load_cache()

    # Apply function to retrieve transcript IDs with caching, ensuring gene matching
    df["ENST_ID"] = df.apply(
        lambda row: (
            str(
                get_best_transcript_id(
                    row["#CHROM"],
                    row["POS"],
                    row["REF"],
                    row["ALT"],
                    row["Gene"],
                    transcript_cache,
                )
                or "None"
            )
            if pd.notna(row["Gene"])
            else "None"
        ),
        axis=1,
    )

    # Display first few rows
    print(df.head())

    # Save updated dataset
    output_file = "./data/nonACMGPLP_pLoF_annotated_transcripts.xlsx"
    df.to_excel(output_file, index=False)

    print(f"✅ Transcript annotations saved to {output_file}!")
