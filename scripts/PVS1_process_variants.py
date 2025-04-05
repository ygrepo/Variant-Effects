import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from esm_variants_utils import (
    load_model,
    get_wt_LLR,
    get_PLLR,
    get_minLLR,
    get_start_loss_LLR,
    compute_delins_llr,
    get_local_PLL,
    align_deletion_pll,
)
import matplotlib.pyplot as plt


# Directory for protein sequences
FASTA_DIR = "./data/protein_sequences/"
VARIANTS_FILE = "./data/variants_processed.xlsx"
OUTPUT_FILE = "./data/variants_with_esm2_llr.xlsx"


# Compute LLR for missense and nonsense mutations
def compute_llr(
    model_type, model, tokenizer, protein_seq, position, ref_aa, alt_aa, device
):
    seq_df = pd.DataFrame(
        [("protein1", "Gene_Name", protein_seq, len(protein_seq))],
        columns=["id", "gene", "seq", "length"],
    )
    _, LLRs = get_wt_LLR(
        seq_df, model_type, model, tokenizer, device=device, silent=True
    )

    if len(LLRs) > 0:
        llr_matrix = LLRs[0]
        return llr_matrix.loc[alt_aa, f"{ref_aa} {position}"]

    return "N/A"


# Compute PLLR for insertions, deletions, frameshifts
def compute_pllr(model_type, model, tokenizer, wt_seq, mut_seq, start_pos, device):
    """
    Compute the Protein Log-Likelihood Ratio (PLLR) for frameshift, insertion, or deletion mutations.
    """
    return get_PLLR(
        wt_seq,
        mut_seq,
        start_pos,
        model_type,
        model,
        tokenizer,
        weighted=False,
        device=device,
    )


# Process Variants for LLR Computation
def run_llr_predictions(model_type, model, tokenizer, device):
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
        print(
            f"Processing {gene} {transcript} {mutation_type}",
            f"{position} {ref_aa} -> {alt_aa}",
        )
        # Compute LLR based on mutation type
        if mutation_type in ["Missense", "Nonsense"]:
            llr = compute_llr(
                model_type,
                model,
                tokenizer,
                protein_seq,
                position,
                ref_aa,
                alt_aa,
                device,
            )
        elif mutation_type == "Insertion":
            mut_seq = protein_seq[:position] + alt_aa + protein_seq[position:]
            llr = compute_pllr(
                model_type, model, tokenizer, protein_seq, mut_seq, position, device
            )
        elif mutation_type == "Deletion":
            mut_seq = protein_seq[: position - 1] + protein_seq[position:]
            llr = compute_pllr(
                model_type, model, tokenizer, protein_seq, mut_seq, position, device
            )
        elif mutation_type == "Stop Loss":
            llr = get_minLLR(protein_seq, position, model, tokenizer, device)
        elif mutation_type == "Start Loss":
            llr = get_start_loss_LLR(protein_seq, model_type, model, tokenizer, device)
        elif mutation_type == "Delins":
            # Ensure the mutation is properly formatted
            if alt_aa == "X":
                if position > len(protein_seq):  # Check if position is valid
                    print(
                        f"Skipping {gene} {transcript} {mutation_type} at {position} (Position out of range)"
                    )
                    llr = "N/A"
                elif protein_seq[position - 1] == "X":  # Stop codon remains unchanged
                    print(
                        f"Skipping {gene} {transcript} {mutation_type} at {position} (Stop codon remains unchanged)"
                    )
                    llr = "N/A"
                else:
                    # Stop codon moves later, compute LLR
                    print(
                        f"Stop codon shifts for {gene} {transcript} {mutation_type} at {position}"
                    )
                    mut_seq = (
                        protein_seq[: position - 1] + alt_aa + protein_seq[position:]
                    )
                    llr = compute_delins_llr(
                        model_type,
                        model,
                        tokenizer,
                        protein_seq,
                        mut_seq,
                        position,
                        alt_aa,
                        device,
                    )
            else:
                # Normal Delins processing
                mut_seq = protein_seq[: position - 1] + alt_aa + protein_seq[position:]
                llr = compute_delins_llr(
                    model_type,
                    model,
                    tokenizer,
                    protein_seq,
                    mut_seq,
                    position,
                    alt_aa,
                    device,
                )
        elif mutation_type == "Frameshift":
            print(f"Frameshift for {gene} {transcript} {mutation_type} at {position}")
            llr = "N/A"
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
    df_results.to_excel(OUTPUT_FILE, index=False)
    print(f"LLR predictions saved to {OUTPUT_FILE}")


def test_llr_insertions(model_type, model, tokenizer, device):
    # Test Insertions
    gene = "MSH6"
    transcript = "ENST00000234420.11"
    fasta_file = os.path.join(FASTA_DIR, f"{gene}_{transcript}.fasta")
    with open(fasta_file, "r") as f:
        protein_seq = "".join(line.strip() for line in f if not line.startswith(">"))
    print(f"Protein Seq.: {protein_seq}\n")
    position = 50
    alt_aa = "G"
    mut_seq = protein_seq[:position] + alt_aa + protein_seq[position:]
    print(f"Mut. Protein Seq.: {mut_seq}\n")
    llr = compute_pllr(
        model_type, model, tokenizer, protein_seq, mut_seq, position, device
    )
    print(f"PLLR (Insertion): {llr}\n")  # esm1: 1.31; esm2: 0.48


def test_llr_delins(model_type, model, tokenizer, device):
    # Test Delins
    gene = "MSH6"
    transcript = "ENST00000540021.6"
    fasta_file = os.path.join(FASTA_DIR, f"{gene}_{transcript}.fasta")
    with open(fasta_file, "r") as f:
        protein_seq = "".join(line.strip() for line in f if not line.startswith(">"))
    print(f"Protein Seq.: {protein_seq}\n")
    position = 10
    alt_aa = "X"
    mut_seq = protein_seq[: position - 1] + alt_aa + protein_seq[position:]
    print(f"Mut. Protein Seq.: {mut_seq}\n")
    llr = compute_delins_llr(
        model_type, model, tokenizer, protein_seq, mut_seq, position, alt_aa, device
    )  # model_type is either esm1 or esm2
    print(f"PLLR (DelInsertion): {llr}\n")  # esm1: 3.15; esm2: 0.95


def test_llr_missense(model_type, model, tokenizer, device):
    # Test Missense
    gene = "MSH6"
    transcript = "ENST00000540021.6"
    fasta_file = os.path.join(FASTA_DIR, f"{gene}_{transcript}.fasta")
    with open(fasta_file, "r") as f:
        protein_seq = "".join(line.strip() for line in f if not line.startswith(">"))
    print(f"Protein Seq.: {protein_seq}\n")

    position = 173
    ref_aa = "M"
    alt_aa = "I"

    llr = compute_llr(
        model_type, model, tokenizer, protein_seq, position, ref_aa, alt_aa, device
    )
    print(f"LLR (Missense): {llr}\n")  # esm1: -2.48, esm2: -2.51

    # Create a DataFrame for the protein sequence
    seq_df = pd.DataFrame(
        [("protein1", "Gene_Name", protein_seq, len(protein_seq))],
        columns=["id", "gene", "seq", "length"],
    )

    # Compute LLRs for the wild-type sequence.
    _, LLRs = get_wt_LLR(
        seq_df, model_type, model, tokenizer, device=device, silent=True
    )

    if len(LLRs) > 0:
        llr_matrix = LLRs[0]
        column_label = f"{ref_aa} {position}"

        # Get the specific LLR for the mutation p.M173I
        mutation_llr = llr_matrix.loc[alt_aa, column_label]
        print(f"LLR for mutation p.{ref_aa}{position}{alt_aa}: {mutation_llr}")

        # Plot the bar chart for all possible substitutions at the given position
        scores = llr_matrix[column_label]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(scores.index, scores.values, color="skyblue")

        # Highlight the bar for the selected alternate amino acid
        for i, aa in enumerate(scores.index):
            if aa == alt_aa:
                bars[i].set_color("red")
                ax.annotate(
                    f"{scores[aa]:.2f}",
                    xy=(i, scores[aa]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    color="red",
                )

        ax.set_title(
            f"LLR Scores for Substitutions at Position {position} (WT: {ref_aa})"
        )
        ax.set_xlabel("Alternate Amino Acid")
        ax.set_ylabel("LLR Score")
        plt.show()
    else:
        print("LLRs not computed.")


def test_deletion(model_type, model, tokenizer, device):
    # Load the protein sequence from a FASTA file.
    gene = "MSH6"
    transcript = "ENST00000540021.6"
    fasta_file = os.path.join(FASTA_DIR, f"{gene}_{transcript}.fasta")

    with open(fasta_file, "r") as f:
        # Skip header lines starting with '>' and join the rest.
        protein_seq = "".join(line.strip() for line in f if not line.startswith(">"))

    print(f"Protein Sequence:\n{protein_seq}\n")

    # Define the mutation: deletion at position 173 (HGVS: p.M173del)
    position = 173
    # For a deletion, remove the amino acid at the specified position.
    mut_seq = protein_seq[: position - 1] + protein_seq[position:]

    print(f"Protein Mutated Sequence:\n{mut_seq}\n")

    # Compute the overall PLLR comparing the wild-type and mutated sequences.
    llr = compute_pllr(
        model_type, model, tokenizer, protein_seq, mut_seq, position, device
    )
    print(
        f"Overall PLLR for deletion at position {position}: {llr}\n"
    )  # esm1: 1.53, esm2: 0.86

    # ---- Obtain Local PLL Values from the Model ----
    # Note: A negative PLL does NOT mean the model thinks the residue is "wrong."
    #       Log probabilities < 1 become negative when taking log.
    wt_local_pll = get_local_PLL(
        protein_seq, model_type, model, tokenizer, position, device
    )
    mut_local_pll = get_local_PLL(
        mut_seq, model_type, model, tokenizer, position, device
    )

    # Because the deletion removes one residue, the mutant PLL array has one fewer element.
    # We'll align the mutant array to the wild-type indices by inserting NaN at the deletion position.
    aligned_mut_pll = align_deletion_pll(wt_local_pll, mut_local_pll, position)

    # ---- Plot 1: Raw PLL Comparison Around the Deletion ----
    window = 10
    start_idx = max(0, position - window - 1)
    end_idx = min(len(wt_local_pll), position + window)

    x_positions = np.arange(start_idx + 1, end_idx + 1)
    wt_window = wt_local_pll[start_idx:end_idx]
    mut_window = aligned_mut_pll[start_idx:end_idx]

    _, ax = plt.subplots(figsize=(10, 6))
    width = 0.35

    ax.bar(
        x_positions - width / 2,
        wt_window,
        width,
        label="Wild-type PLL",
        color="skyblue",
    )
    ax.bar(
        x_positions + width / 2,
        mut_window,
        width,
        label="Mutant PLL (Deletion)",
        color="orange",
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Local PLL Value")
    ax.set_title("Local PLL Comparison Around Deletion (p.M173del) - Raw")
    ax.axvline(position, color="red", linestyle="--", label="Deletion Position")
    ax.legend()
    plt.show()

    # ---- Plot 2: Normalized PLL (Difference from Wild Type) ----
    # Here we shift the wild-type PLL to zero and measure how the mutant differs.
    # We first compute the difference array: (Mutant PLL - Wild-type PLL).
    # We'll align that difference array in the same way as above (inserting NaN).

    # Compute difference for positions that exist in both arrays (i.e., ignoring the alignment step).
    mut_diff = mut_local_pll - wt_local_pll[: len(mut_local_pll)]
    aligned_mut_diff = align_deletion_pll(wt_local_pll, mut_diff, position)

    mut_diff_window = aligned_mut_diff[start_idx:end_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x_positions,
        mut_diff_window,
        width,
        color="purple",
        label="Mutant PLL - Wild-type PLL",
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("ΔPLL (Mut - WT)")
    ax.set_title("Local PLL Difference from Wild Type Around Deletion (p.M173del)")
    ax.axvline(position, color="red", linestyle="--", label="Deletion Position")
    ax.legend()
    plt.show()


# Run the pipeline
if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    # Load ESM Model
    # model_name = "facebook/esm1b_t33_650M_UR50S"
    model_name = "facebook/esm2_t6_8M_UR50D"
    model, tokenizer, device = load_model(model_name=model_name, load_flag=True)
    print(f"Device: {device}")
    print(f"Model: {model_name} loaded successfully!")
    # test_llr_insertions("esm1", model, tokenizer, device)
    # test_llr_delins("esm1", model, tokenizer, device)
    # test_deletion("esm1", model, tokenizer, device)
    # test_llr_missense("esm2", model, tokenizer, device)
    run_llr_predictions("esm2", model, tokenizer, device)
