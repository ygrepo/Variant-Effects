import os
import csv
import torch
import requests
import pandas as pd
import re
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EsmForMaskedLM,
)


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


def fetch_uniprot_sequence(uniprot_id):
    """Fetches the protein sequence from UniProt given a UniProt ID."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        return "".join(fasta_data.split("\n")[1:])
    else:
        print(f"Failed to fetch sequence for {uniprot_id}")
        return None


def compute_llr(model, tokenizer, input_ids, position_1based, ref_aa, alt_aa):
    """Computes log-likelihood ratio for a variant."""
    idx_in_input = position_1based
    masked_input = input_ids.clone()
    masked_input[0, idx_in_input] = tokenizer.mask_token_id

    with torch.no_grad():
        logits = model(masked_input).logits

    probs = torch.nn.functional.softmax(logits[0, idx_in_input], dim=0)
    log_probs = torch.log(probs)

    ref_token_id = tokenizer.convert_tokens_to_ids(ref_aa)
    alt_token_id = tokenizer.convert_tokens_to_ids(alt_aa)

    return log_probs[alt_token_id].item() - log_probs[ref_token_id].item()


def load_protein_sequence_from_fasta(fasta_path):
    """Loads a protein sequence from a saved FASTA file."""
    if not os.path.exists(fasta_path):
        return None

    with open(fasta_path, "r") as f:
        lines = f.readlines()

    return "".join(line.strip() for line in lines if not line.startswith(">"))


def run_predictions(df, model, tokenizer, device, fasta_dir="./data/protein_sequences"):
    """Predicts LLR for all variants, fetching sequences only if they are not saved."""
    os.makedirs(fasta_dir, exist_ok=True)  # Ensure the output directory exists
    results = []

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
                    f"⚠️ Failed to retrieve sequence for {gene} ({transcript}), skipping..."
                )
                continue  # Skip this transcript if no sequence is available

        # Encode protein sequence for the model
        input_ids = tokenizer.encode(protein_seq, return_tensors="pt").to(device)

        # Process variants for this (Gene, Transcript) pair
        transcript_variants = df[
            (df["Gene"] == gene) & (df["Transcript"] == transcript)
        ]
        for _, var in transcript_variants.iterrows():
            parsed = parse_hgvsp(var["HGVSp"])
            if not parsed:
                print(f"⚠️ Warning: Invalid HGVSp notation: {var['HGVSp']}, skipping...")
                continue

            prot_pos, ref_aa, alt_aa = parsed
            print(
                f"Processing variant: {gene} ({transcript}) {prot_pos} {ref_aa} -> {alt_aa}"
            )

            # **Variant Validation**
            if "ins" in var["HGVSp"]:  # **Insertion validation**
                if (prot_pos - 1 < len(protein_seq)) and (prot_pos < len(protein_seq)):
                    if (
                        protein_seq[prot_pos - 1] == ref_aa[0]
                        and protein_seq[prot_pos] == ref_aa[1]
                    ):
                        print(
                            f"✅ Valid Insertion: {gene} at {prot_pos}, {ref_aa} -> {alt_aa}"
                        )
                    else:
                        print(
                            f"❌ Invalid Insertion: Expected {ref_aa} at {prot_pos}, found {protein_seq[prot_pos - 1:prot_pos+1]}. Skipping..."
                        )
                        continue
                else:
                    print(
                        f"❌ Index out of range: {prot_pos} exceeds sequence length ({len(protein_seq)}). Skipping..."
                    )
                    continue

            elif "delins" in var["HGVSp"]:  # **Delins validation**
                if protein_seq[prot_pos - 1] == ref_aa:
                    print(
                        f"✅ Valid Delins: {gene} at {prot_pos}, {ref_aa} -> {alt_aa}"
                    )
                else:
                    print(
                        f"❌ Invalid Delins: Expected {ref_aa} at {prot_pos}, found {protein_seq[prot_pos - 1]}. Skipping..."
                    )
                    continue

            elif "del" in var["HGVSp"]:  # **Deletion validation**
                if protein_seq[prot_pos - 1] == ref_aa:
                    print(
                        f"✅ Valid Deletion: {gene} at {prot_pos}, {ref_aa} -> {alt_aa}"
                    )
                else:
                    print(
                        f"❌ Invalid Deletion: Expected {ref_aa} at {prot_pos}, found {protein_seq[prot_pos - 1]}. Skipping..."
                    )
                    continue

            elif (
                "fs" in var["HGVSp"] or "?" in var["HGVSp"] or "X" in var["HGVSp"]
            ):  # **Frameshift/Nonsense case**
                print(f"⏩ Skipping frameshift/nonsense mutation: {var['HGVSp']}")
                continue

            else:  # **Substitution validation**
                if protein_seq[prot_pos - 1] == ref_aa:
                    print(
                        f"✅ Valid Substitution: {gene} at {prot_pos}, {ref_aa} -> {alt_aa}"
                    )
                else:
                    print(
                        f"❌ Invalid Substitution: Expected {ref_aa} at {prot_pos}, found {protein_seq[prot_pos - 1]}. Skipping..."
                    )
                    continue

            # Compute LLR
            llr = compute_llr(model, tokenizer, input_ids, prot_pos, ref_aa, alt_aa)
            results.append(
                {
                    "Gene": gene,
                    "Transcript": transcript,
                    "HGVSp": var["HGVSp"],
                    "ProteinPos": prot_pos,
                    "RefAA": ref_aa,
                    "AltAA": alt_aa,
                    "LLR": llr,
                }
            )

    return results


def save_results(results, output_file):
    """Saves LLR predictions to CSV."""
    fieldnames = ["Gene", "Transcript", "HGVSp", "ProteinPos", "RefAA", "AltAA", "LLR"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_file}")


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
    print(f"Current directory: {os.getcwd()}")

    # 1) Load dataset
    file_path = "./data/test_Predictions.xlsx"
    #    file_path = "./data/nonACMGPLP_pLoF_allinfo_AA_simplified_with_UniProt_IDs.xlsx"
    df = pd.read_excel(
        file_path, dtype={"#CHROM": str, "REF": str, "ALT": str, "Gene": str}
    )

    # 2) Load model
    model_name = "facebook/esm2_t6_8M_UR50D"  # Or "togethercomputer/evo-1-131k-base"
    model, tokenizer, device = load_model(model_name)

    # 3) Run predictions
    results = run_predictions(df, model, tokenizer, device)

    # 4) Save results
    output_file = "./data/variant_predictions.csv"
    save_results(results, output_file)
