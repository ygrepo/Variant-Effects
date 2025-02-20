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


def parse_protein_consequence(consequence_str):
    """
    Parses HGVS protein notation into (position, ref_aa, alt_aa).

    Examples:
    - 'p.Phe1296Leu' -> (1296, 'F', 'L')
    - 'p.W50_S51insG' -> (50, 'WS', 'WSG')
    - 'p.M1?' -> (1, 'M', '?')
    - 'p.Q66Pfs' -> None (Frameshift mutation)
    - 'p.X95L' -> None (Stop codon mutation)

    Returns:
        (position, ref_aa, alt_aa) or None if parsing fails.
    """

    # Single amino acid substitutions (e.g., p.M173I)
    match = re.match(r"^p\.([A-Za-z]+)(\d+)([A-Za-z?]+)$", consequence_str)
    if match:
        ref_aa_3, pos_str, alt_aa_3 = match.groups()
        pos = int(pos_str)

        ref_aa_1 = three_to_one.get(ref_aa_3, None)
        alt_aa_1 = three_to_one.get(alt_aa_3, None)

        if ref_aa_1 and alt_aa_1:
            return pos, ref_aa_1, alt_aa_1
        return None

    # Insertions (e.g., p.W50_S51insG or p.E47_L48insSGPEE)
    match = re.match(
        r"^p\.([A-Za-z]+)(\d+)_([A-Za-z]+)(\d+)ins([A-Za-z]+)$", consequence_str
    )
    if match:
        ref_start_aa_3, pos_start_str, ref_end_aa_3, pos_end_str, inserted_aa_3 = (
            match.groups()
        )
        pos_start = int(pos_start_str)
        pos_end = int(pos_end_str)

        ref_start_aa_1 = three_to_one.get(ref_start_aa_3, None)
        ref_end_aa_1 = three_to_one.get(ref_end_aa_3, None)
        inserted_aa_1 = "".join(
            [
                three_to_one.get(aa, "")
                for aa in re.findall(r"[A-Z][a-z]{2}", inserted_aa_3)
            ]
        )

        if ref_start_aa_1 and ref_end_aa_1 and inserted_aa_1:
            return (
                pos_start,
                ref_start_aa_1 + ref_end_aa_1,
                ref_start_aa_1 + ref_end_aa_1 + inserted_aa_1,
            )
        return None

    # Frameshift or nonsense mutations (e.g., p.Q66Pfs, p.X95L, p.M1?)
    if "fs" in consequence_str or "?" in consequence_str or "X" in consequence_str:
        return None  # Skip frameshifts and nonsense mutations

    return None  # If none of the patterns match


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
    unique_genes = df[["Gene", "UniProt_ID"]].drop_duplicates()
    print(f"Unique genes: {len(unique_genes)}")

    # Iterate over unique genes and fetch protein sequences if necessary
    for _, row in unique_genes.iterrows():
        gene = row["Gene"]
        uniprot_id = row["UniProt_ID"]
        fasta_filename = os.path.join(fasta_dir, f"{gene}_{uniprot_id}.fasta")

        # Try loading from saved FASTA first
        protein_seq = load_protein_sequence_from_fasta(fasta_filename)
        print(f"Loaded sequence for {gene} ({uniprot_id}) from {fasta_filename}")

        if not protein_seq:
            print(f"Fetching sequence for {gene} ({uniprot_id}) from UniProt...")
            protein_seq = fetch_uniprot_sequence(uniprot_id)

            if protein_seq:
                # Save the sequence for future use
                with open(fasta_filename, "w") as f:
                    f.write(f">{gene}_{uniprot_id}\n{protein_seq}\n")
                print(f"Saved sequence: {fasta_filename}")
            else:
                print(
                    f"Failed to retrieve sequence for {gene} ({uniprot_id}), skipping..."
                )
                continue  # Skip this gene if no sequence is available

        # Encode protein sequence for the model
        input_ids = tokenizer.encode(protein_seq, return_tensors="pt").to(device)

        # Process variants for this gene
        gene_variants = df[df["Gene"] == gene]
        for _, var in gene_variants.iterrows():
            parsed = parse_protein_consequence(var["HGVSp"])
            if not parsed:
                continue

            prot_pos, ref_aa, alt_aa = parsed

            print(f"Processing variant: {gene} {prot_pos} {ref_aa} -> {alt_aa}")
            # Validate sequence match
            if protein_seq[prot_pos - 1] != ref_aa:
                print(
                    f"Warning: Mismatch at position {prot_pos} in {gene}, skipping..."
                )
                continue

            # Compute LLR
            llr = compute_llr(model, tokenizer, input_ids, prot_pos, ref_aa, alt_aa)
            results.append(
                {
                    "Gene": gene,
                    "UniProt_ID": uniprot_id,
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
    fieldnames = ["Gene", "UniProt_ID", "HGVSp", "ProteinPos", "RefAA", "AltAA", "LLR"]
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
    file_path = "./data/nonACMGPLP_pLoF_allinfo_AA_simplified_with_UniProt_IDs.xlsx"
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
    # save_results(results, output_file)
