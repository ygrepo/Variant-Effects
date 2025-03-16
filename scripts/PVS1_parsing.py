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

    Fixes:
    - Frameshift mutations (`p.Q66Pfs`, `p.T229Afs`) without explicit stop codons.
    - Ensures frameshifts are properly classified even when stop codon information is missing.

    Returns:
        tuple: (position, ref_aa, alt_aa, mutation_type)
    """

    # Start Codon Loss (e.g., p.M1?, p.M1V)
    match = re.match(r"^p\.M1(\?|[A-Z])$", hgvsp_str)
    if match:
        alt_aa = match.group(1)
        return (1, "M", alt_aa, "Start Loss")

    # Frameshift Mutations (e.g., p.Q66Pfs*5, p.Q66Pfs)
    match = re.match(r"^p\.([A-Z])(\d+)([A-Z?])fs(\*\d+)?$", hgvsp_str)
    if match:
        ref_aa, pos, alt_aa, stop = match.groups()
        stop = stop if stop else "*?"  # If stop codon is missing, set it to *?
        return (int(pos), ref_aa, f"{alt_aa}fs{stop}", "Frameshift")

    # Stop Codon Loss (e.g., p.X104L)
    match = re.match(r"^p\.X(\d+)([A-Z])$", hgvsp_str)
    if match:
        pos, alt_aa = match.groups()
        return (int(pos), "X", alt_aa, "Stop Loss")

    # Nonsense Mutations (Stop Gain, e.g., p.R104X)
    match = re.match(r"^p\.([A-Z])(\d+)X$", hgvsp_str)
    if match:
        ref_aa, pos = match.groups()
        return (int(pos), ref_aa, "X", "Nonsense")

    # Missense Mutations (e.g., p.Y97C)
    match = re.match(r"^p\.([A-Z])(\d+)([A-Z])$", hgvsp_str)
    if match:
        ref_aa, pos, alt_aa = match.groups()
        return (int(pos), ref_aa, alt_aa, "Missense")

    # Insertions (e.g., p.W50_S51insG)
    match = re.match(r"^p\.([A-Z]?)(\d+)_([A-Z]?)(\d+)ins([A-Z]+)$", hgvsp_str)
    if match:
        ref_start_aa, pos_start, ref_end_aa, pos_end, inserted_aa = match.groups()
        pos_start, pos_end = int(pos_start), int(pos_end)
        return (pos_start, ref_start_aa + ref_end_aa, inserted_aa, "Insertion")

    # Deletions (e.g., p.A123del)
    match = re.match(r"^p\.([A-Z])(\d+)del$", hgvsp_str)
    if match:
        ref_aa, pos = match.groups()
        return (int(pos), ref_aa, "-", "Deletion")

    # Deletion-Insertion (Delins, e.g., p.X1231delinsX)
    match = re.match(r"^p\.([A-Z])(\d+)delins([A-Z]+)$", hgvsp_str)
    if match:
        ref_aa, pos, alt_aa = match.groups()
        return (int(pos), ref_aa, alt_aa, "Delins")

    return None  # Unrecognized format


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
    print(f"Url: {url}")

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


def parse(
    df,
    fasta_dir="./data/protein_sequences",
    out_file="./data/variants_processed.csv",
):
    """Predicts LLR for all variants, fetching sequences only if they are not saved."""
    os.makedirs(fasta_dir, exist_ok=True)  # Ensure the output directory exists
    # Ensure we process by (Gene, Transcript)
    unique_gene_transcripts = df[["Gene", "Transcript"]].drop_duplicates()
    print(f"Unique (Gene, Transcript) pairs: {len(unique_gene_transcripts)}")

    # Iterate over unique (Gene, Transcript) pairs
    processed_variants = []
    for _, row in unique_gene_transcripts.iterrows():
        gene = row["Gene"]
        transcript = row["Transcript"]
        transcript_variants = df[
            (df["Gene"] == gene) & (df["Transcript"] == transcript)
        ]

        for _, var in transcript_variants.iterrows():
            print(f"Processing: {var['HGVSp']}")
            parsed_hvsp = parse_hgvsp(var["HGVSp"])
            if not parsed_hvsp:
                print(f"Warning: Invalid HGVSp notation: {var['HGVSp']}, skipping...")
                continue
            position, ref_aa, alt_aa, mutation_type = parsed_hvsp
            print(
                f"Processing variant: {gene}-{mutation_type}-{transcript}) {position} {ref_aa} -> {alt_aa}"
            )
            # Append to the processed list
            processed_variants.append(
                {
                    "Gene": gene,
                    "Transcript": transcript,
                    "HGVSp": var["HGVSp"],
                    "MutationType": mutation_type,
                    "Position": position,
                    "RefAA": ref_aa,
                    "AltAA": alt_aa,
                }
            )

    processed_df = pd.DataFrame(processed_variants)

    # Save to CSV file
    processed_df.to_csv(out_file, index=False)
    print(f"✅ Processed variants saved to {out_file}")


def load_protein_files(df, fasta_dir="./data/protein_sequences"):
    """Fetches and validates protein sequences for all variants."""
    os.makedirs(fasta_dir, exist_ok=True)  # Ensure the output directory exists

    # Process by (Gene, Transcript)
    unique_gene_transcripts = df[["Gene", "Transcript"]].drop_duplicates()
    print(f"Unique (Gene, Transcript) pairs: {len(unique_gene_transcripts)}")

    for _, row in unique_gene_transcripts.iterrows():
        gene = row["Gene"]
        transcript = row["Transcript"]
        fasta_filename = os.path.join(fasta_dir, f"{gene}_{transcript}.fasta")

        # Load from saved FASTA first
        protein_seq = load_protein_sequence_from_fasta(fasta_filename)
        if protein_seq:
            print(f"Loaded sequence for {gene} ({transcript}) from {fasta_filename}")
        else:
            print(f"Fetching sequence for {gene} ({transcript}) from Ensembl...")
            protein_seq = fetch_transcript_protein_sequence(transcript)

            if protein_seq:
                # Save for future use
                with open(fasta_filename, "w") as f:
                    f.write(f">{gene}_{transcript}\n{protein_seq}\n")
                print(f"Saved sequence: {fasta_filename}")
            else:
                print(
                    f"Failed to retrieve sequence for {gene} ({transcript}), skipping..."
                )
                continue  # Skip if no sequence is available


if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    # print(parse_hgvsp("p.M1?"))  # Start Codon Loss (Unknown Replacement)
    # print(parse_hgvsp("p.M1V"))  # Start Codon Loss (M1 to V)
    # print(parse_hgvsp("p.X104L"))  # Stop Codon Loss
    # print(parse_hgvsp("p.R104X"))  # Nonsense Mutation (Stop Gain)
    # print(parse_hgvsp("p.Y97C"))  # Missense Mutation
    # print(parse_hgvsp("p.A100Dfs*5"))  # Frameshift Mutation
    # print(parse_hgvsp("p.W50_S51insG"))  # Insertion
    # print(parse_hgvsp("p.A123del"))  # Deletion
    # print(parse_hgvsp("p.X1231delinsX"))  # Delins

    # print(parse_hgvsp("p.M1?"))  # Start Codon Loss (Unknown Replacement)
    # print(parse_hgvsp("p.M71V"))  # Missense Substitution
    # print(parse_hgvsp("p.X104L"))  # Stop Codon Loss
    # print(parse_hgvsp("p.R104X"))  # Nonsense Mutation (Stop Gain)
    # print(parse_hgvsp("p.W50_S51insG"))  # Insertion
    # print(parse_hgvsp("p.A123del"))  # Deletion

    # Load dataset
    # file_path = "./data/nonACMGPLP_pLoF_allinfo_AA_Light.xlsx"
    # df = pd.read_excel(
    #     file_path, dtype={"#CHROM": str, "REF": str, "ALT": str, "Gene": str}
    # )
    # parse(df)

    file_path = "./data/variants_processed.csv"
    df = pd.read_csv(file_path)
    print(df.head())
    # Validate variants
    load_protein_files(df)
