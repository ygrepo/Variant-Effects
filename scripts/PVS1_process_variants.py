import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from esm_variants_utils import (
    get_batch_converter,
    load_model,
    get_wt_LLR,
    get_PLLR,
    get_minLLR,
)


# Directory for protein sequences
FASTA_DIR = "./data/protein_sequences/"
VARIANTS_FILE = "./data/variants_processed.csv"
OUTPUT_FILE = "./data/variants_with_llr.csv"


# Compute LLR for missense and nonsense mutations
def compute_llr(model, tokenizer, protein_seq, position, ref_aa, alt_aa, device):
    seq_df = pd.DataFrame(
        [("protein1", "Gene_Name", protein_seq, len(protein_seq))],
        columns=["id", "gene", "seq", "length"],
    )
    batch_converter = get_batch_converter(tokenizer, device, protein_seq)
    input_df_ids, LLRs = get_wt_LLR(
        seq_df, model, tokenizer, batch_converter, device=device, silent=True
    )

    if len(LLRs) > 0:
        llr_matrix = LLRs[0]
        return llr_matrix.loc[alt_aa, f"{ref_aa} {position}"]

    return "N/A"


# Compute PLLR for insertions, deletions, frameshifts
def compute_pllr(model, tokenizer, wt_seq, mut_seq, start_pos, device):
    """
    Compute the Protein Log-Likelihood Ratio (PLLR) for frameshift, insertion, or deletion mutations.
    """
    batch_converter = get_batch_converter(tokenizer, device, wt_seq)
    return get_PLLR(
        wt_seq,
        mut_seq,
        start_pos,
        model,
        tokenizer,
        batch_converter,
        weighted=False,
        device=device,
    )


# Compute LLR for stop-gain mutations
# Compute LLR for stop-loss mutations using Hugging Face ESM
def compute_min_llr(seq, stop_pos, model, tokenizer, batch_converter, device):
    """
    Compute the minimum LLR for stop-loss mutations.
    """
    return get_minLLR(seq, stop_pos, model, tokenizer, batch_converter, device)


# Process Variants for LLR Computation
def run_llr_predictions(model, tokenizer, batch_converter, device):
    if not os.path.exists(VARIANTS_FILE):
        print(f"Error: Variants file {VARIANTS_FILE} not found!")
        return

    df = pd.read_excel(VARIANTS_FILE)
    print(f"Loaded {len(df)} variants from {VARIANTS_FILE}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing LLR"):
        gene = row["Gene"]
        transcript = row["Transcript"]
        fasta_file = os.path.join(FASTA_DIR, f"{gene}_{transcript}.fasta")

        # Load protein sequence
        if not os.path.exists(fasta_file):
            print(f"Error: Missing protein sequence file {fasta_file}, skipping...")
            continue

        with open(fasta_file, "r") as f:
            protein_seq = "".join(
                line.strip() for line in f if not line.startswith(">")
            )

        # Mutation details
        position = int(row["Position"])
        ref_aa = row["RefAA"]
        alt_aa = row["AltAA"]
        mutation_type = row["MutationType"]

        # Compute LLR based on mutation type
        if mutation_type in ["Missense", "Nonsense"]:
            llr = compute_llr(
                model, tokenizer, protein_seq, position, ref_aa, alt_aa, device
            )
        elif mutation_type in ["Frameshift", "Insertion", "Deletion"]:
            mut_seq = (
                protein_seq[: position - 1] + alt_aa + protein_seq[position:]
            )  # Simulated mutation
            llr = compute_pllr(model, tokenizer, protein_seq, mut_seq, position, device)
        elif mutation_type == "Stop Loss":
            llr = compute_min_llr(
                protein_seq, position, model, tokenizer, batch_converter, device
            )
        else:
            llr = "N/A"

        # Store results
        results.append(
            {
                "Gene": gene,
                "Transcript": transcript,
                "HGVSp": row["HGVSp"],
                "MutationType": mutation_type,
                "Position": position,
                "RefAA": ref_aa,
                "AltAA": alt_aa,
                "LLR": llr,
            }
        )

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"LLR predictions saved to {OUTPUT_FILE}")


# Run the pipeline
if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")

    # Load ESM Model
    model, tokenizer, device = load_model("facebook/esm2_t6_8M_UR50D")
    run_llr_predictions()
