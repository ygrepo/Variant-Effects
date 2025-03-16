import os
import pandas as pd
import re
import requests

# Mapping 3-letter amino acids to 1-letter
three_to_one = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Xaa": "X",
    "Ter": "*",
}


def parse_hgvsp(hgvsp_str):
    """
    Parses HGVS protein notation into a structured format.

    Handles:
    - Substitutions: p.M173I -> (173, 'M', 'I')
    - Insertions: p.W50_S51insG -> (50, 'WS', 'WSG')
    - Deletions: p.A123del -> (123, 'A', '-')
    - Deletion-Insertion (delins): p.X1231delinsX -> (1231, 'X', 'X')
    - Frameshift & Nonsense Mutations -> Ignored (returns None)

    Returns:
        tuple: (position, ref_aa, alt_aa) or None if not applicable
    """

    # Single amino acid substitutions (e.g., p.M173I)
    print(f"hgvsp_str: {hgvsp_str}")
    match = re.match(r"^p\.([A-Z])(\d+)([A-Z?])$", hgvsp_str)
    if match:
        ref_aa, pos, alt_aa = match.groups()
        return int(pos), ref_aa, alt_aa

    # Insertions (e.g., p.W50_S51insG or p.E47_L48insSGPEE)
    # Insertions (e.g., p.W50_S51insG)
    match = re.match(r"^p\.([A-Z])(\d+)_([A-Z])(\d+)ins([A-Z]+)$", hgvsp_str)
    if match:
        ref_start_aa, pos_start, ref_end_aa, pos_end, inserted_aa = match.groups()
        pos_start, pos_end = int(pos_start), int(pos_end)

        return pos_start, ref_start_aa + ref_end_aa, ref_start_aa + inserted_aa

    # Deletions (e.g., p.A123del)
    match = re.match(r"^p\.([A-Z])(\d+)del$", hgvsp_str)
    if match:
        ref_aa, pos = match.groups()
        return int(pos), ref_aa, "-"

    # Deletion-Insertion (e.g., p.X1231delinsX)
    match = re.match(r"^p\.([A-Z])(\d+)delins([A-Z]+)$", hgvsp_str)
    if match:
        ref_aa, pos, alt_aa = match.groups()
        return int(pos), ref_aa, alt_aa

    # Frameshift or nonsense mutations (ignored)
    if "fs" in hgvsp_str or "?" in hgvsp_str or "X" in hgvsp_str:
        return None

    return None  # If none of the patterns match


def fetch_transcript_protein_sequence(transcript_id):
    """Fetches the protein sequence from Ensembl given a transcript ID (ENST_...).

    - First tries the exact transcript version (e.g., ENST00000256474.3).
    - If not found, retries with the unversioned transcript (ENST00000256474).

    Args:
        transcript_id (str): Ensembl transcript ID (e.g., ENST00000256474.3).

    Returns:
        str: Protein sequence as a string or None if not found.
    """
    url = f"https://rest.ensembl.org/sequence/id/{transcript_id}?type=protein&content-type=text/plain"

    response = requests.get(url)

    if response.status_code == 200:
        return response.text.strip()  # Return protein sequence

    # If the exact version is not found, try without the version number
    unversioned_transcript = transcript_id.split(".")[0]  # Remove version suffix
    if unversioned_transcript != transcript_id:
        print(
            f"Transcript {transcript_id} not found, retrying with {unversioned_transcript}..."
        )
        return fetch_transcript_protein_sequence(unversioned_transcript)

    print(f"Failed to fetch sequence for transcript {transcript_id}.")
    return None


def load_protein_sequence_from_fasta(fasta_path):
    """Loads a protein sequence from a saved FASTA file."""
    if not os.path.exists(fasta_path):
        return None

    with open(fasta_path, "r") as f:
        lines = f.readlines()

    return "".join(line.strip() for line in lines if not line.startswith(">"))


def validate_hgvsp(parsed_hvsp, protein_seq, gene, transcript):
    """
    Validates the hgvsp data to ensure it meets certain criteria.
    """
    prot_pos, ref_aa, alt_aa = parsed_hvsp
    print(f"Processing variant: {gene} ({transcript}) {prot_pos} {ref_aa} -> {alt_aa}")

    # **Variant Validation**
    if "ins" in parsed_hvsp:  # **Insertion validation**
        if (prot_pos - 1 < len(protein_seq)) and (prot_pos < len(protein_seq)):
            if (
                protein_seq[prot_pos - 1] == ref_aa[0]
                and protein_seq[prot_pos] == ref_aa[1]
            ):
                print(f"Valid Insertion: {gene} at {prot_pos}, {ref_aa} -> {alt_aa}")
            else:
                print(
                    f"Invalid Insertion: Expected {ref_aa} at {prot_pos}. Skipping..."
                )
                return
        else:
            print(f"Out of range: {prot_pos} > {len(protein_seq)}. Skipping...")
            return

    elif "delins" in parsed_hvsp:  # **Delins validation**
        if protein_seq[prot_pos - 1] == ref_aa:
            print(f"Valid Delins: {gene} at {prot_pos}, {ref_aa} -> {alt_aa}")
        else:
            print(f"Invalid Delins: Expected {ref_aa} at {prot_pos}. Skipping...")
            return

    elif "del" in parsed_hvsp:  # **Deletion validation**
        if protein_seq[prot_pos - 1] == ref_aa:
            print(f"Valid Deletion: {gene} at {prot_pos}, {ref_aa} -> {alt_aa}")
        else:
            print(f"Invalid Deletion: Expected {ref_aa} at {prot_pos}. Skipping...")
            return

    elif (
        "fs" in parsed_hvsp or "?" in parsed_hvsp or "X" in parsed_hvsp
    ):  # **Frameshift/Nonsense**
        print(f"Skipping frameshift/nonsense mutation: {parsed_hvsp}")
        return

    else:  # **Substitution validation**
        if protein_seq[prot_pos - 1] == ref_aa:
            print(f"Valid Substitution: {gene} at {prot_pos}, {ref_aa} -> {alt_aa}")
        else:
            print(f"Invalid Substitution: Expected {ref_aa} at {prot_pos}. Skipping...")
            return


def parse(df, fasta_dir="./data/protein_sequences"):
    """Predicts LLR for all variants, fetching sequences only if they are not saved."""
    os.makedirs(fasta_dir, exist_ok=True)  # Ensure the output directory exists
    # Ensure we process by (Gene, Transcript)
    unique_gene_transcripts = df[["Gene", "Transcript"]].drop_duplicates()
    print(f"Unique (Gene, Transcript) pairs: {len(unique_gene_transcripts)}")

    # Iterate over unique (Gene, Transcript) pairs
    for _, row in unique_gene_transcripts.iterrows():
        gene = row["Gene"]
        transcript = row["Transcript"]

        fasta_filename = os.path.join(fasta_dir, f"{gene}_{transcript}.fasta")

        # Try loading from saved FASTA first
        protein_seq = load_protein_sequence_from_fasta(fasta_filename)
        if protein_seq:
            print(f"Loaded sequence for {gene} ({transcript}) from {fasta_filename}")
        else:
            print(
                f"Fetching sequence for {gene} ({transcript}) from UniProt or Ensembl..."
            )
            protein_seq = fetch_transcript_protein_sequence(
                transcript
            )  # Updated to fetch by transcript

            if protein_seq:
                # Save the sequence for future use
                with open(fasta_filename, "w") as f:
                    f.write(f">{gene}_{transcript}\n{protein_seq}\n")
                print(f"Saved sequence: {fasta_filename}")
            else:
                print(
                    f"Failed to retrieve sequence for {gene} ({transcript}), skipping..."
                )
                continue  # Skip this transcript if no sequence is available

        # Process variants for this (Gene, Transcript) pair
        transcript_variants = df[
            (df["Gene"] == gene) & (df["Transcript"] == transcript)
        ]
        for _, var in transcript_variants.iterrows():
            parsed_hvsp = parse_hgvsp(var["HGVSp"])
            if not parsed_hvsp:
                print(f"Warning: Invalid HGVSp notation: {var['HGVSp']}, skipping...")
                continue

            validate_hgvsp(parsed_hvsp, protein_seq, gene, transcript)

    # # Ensure we process by (Gene, Transcript)
    # # Ensure we process by (Gene, Transcript)
    # unique_gene_transcripts = df[["Gene", "Transcript"]].drop_duplicates()
    # print(f"Unique (Gene, Transcript) pairs: {len(unique_gene_transcripts)}")
    # protein_sequences = {}

    # # Iterate over unique (Gene, Transcript) pairs
    # for _, row in unique_gene_transcripts.iterrows():
    #     gene, transcript, hgvsp = row["Gene"], row["Transcript"], row.get("HGVSp", "")

    #     if transcript not in protein_sequences:
    #         print(
    #             f"Skipping protein_sequences {transcript} for gene {gene} - no sequence found."
    #         )
    #         continue  # Skip if no sequence is found

    #     protein_seq = protein_sequences[transcript]
    #     parsed_hvsp = parse_hgvsp(hgvsp)
    #     validate_hgvsp(parsed_hvsp, protein_seq, gene, transcript)


if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")

    # 1) Load dataset
    file_path = "./data/nonACMGPLP_pLoF_allinfo_AA_Light.xlsx"
    df = pd.read_excel(
        file_path, dtype={"#CHROM": str, "REF": str, "ALT": str, "Gene": str}
    )
    parse(df)
