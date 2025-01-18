import os
from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import csv


import re

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
    # If you encounter 'Ter' or 'Stop' it indicates nonsense.
}


def parse_protein_consequence(consequence_str):
    """
    Parses strings like 'p.Phe1296Leu' -> (1296, 'F', 'L')
    Returns None if it fails to parse.
    """
    # Regex to capture p.(RefAA)(Position)(AltAA), e.g. p.Phe1296Leu
    match = re.match(r"^p\.([A-Za-z]+)(\d+)([A-Za-z]+)$", consequence_str)
    if not match:
        return None

    ref_aa_3, pos_str, alt_aa_3 = match.groups()
    pos = int(pos_str)

    # Convert the 3-letter to 1-letter
    ref_aa_1 = three_to_one.get(ref_aa_3, None)
    alt_aa_1 = three_to_one.get(alt_aa_3, None)
    if not ref_aa_1 or not alt_aa_1:
        # Could not map (e.g. it's a 'Ter' or something else)
        return None

    return pos, ref_aa_1, alt_aa_1


# Load the model and tokenizer
def compute_llr(model, tokenizer, input_ids, position_1based, ref_aa, alt_aa):
    """
    Masks the position_1based in the input_ids (the protein sequence)
    and returns log_prob(alt_aa) - log_prob(ref_aa).
    """
    # The tokenizer adds special tokens, so typically:
    idx_in_input = position_1based

    masked_input = input_ids.clone()
    masked_input[0, idx_in_input] = tokenizer.mask_token_id

    with torch.no_grad():
        logits = model(masked_input).logits

    probs = torch.nn.functional.softmax(logits[0, idx_in_input], dim=0)
    log_probs = torch.log(probs)

    ref_token_id = tokenizer.convert_tokens_to_ids(ref_aa)
    alt_token_id = tokenizer.convert_tokens_to_ids(alt_aa)

    log_prob_ref = log_probs[ref_token_id].item()
    log_prob_alt = log_probs[alt_token_id].item()

    return log_prob_alt - log_prob_ref


def run_predictions(protein_seq, rare_variants, model, tokenizer, device):
    input_ids = tokenizer.encode(protein_seq, return_tensors="pt").to(device)

    results = []
    for var in rare_variants:
        # e.g. "p.Phe1296Leu"
        prot_consequence_str = var["HGVSp"]

        # Parse out (protein_position, refAA, altAA)
        parsed = parse_protein_consequence(prot_consequence_str)
        if not parsed:
            # If parse fails or is nonsense/frameshift, skip
            continue

        prot_pos, ref_aa, alt_aa = parsed

        # Check that protein_seq indeed has ref_aa at that position
        if protein_seq[prot_pos - 1] != ref_aa:
            print(
                f"Warning: mismatch at protein position {prot_pos}: "
                f"seq={protein_seq[prot_pos - 1]}, var={ref_aa}"
            )
            continue

        # Compute log-likelihood ratio
        llr = compute_llr(model, tokenizer, input_ids, prot_pos, ref_aa, alt_aa)

        results.append(
            {
                "ProteinPos": prot_pos,
                "RefAA": ref_aa,
                "AltAA": alt_aa,
                "LLR": llr,
                # you can also store AF or other fields from var if you want
            }
        )

    # Sort by LLR descending
    results.sort(key=lambda x: x["LLR"], reverse=True)
    return results


def load_protein_sequence(fasta_path):
    lines = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip FASTA header lines starting with '>'
            if line.startswith(">"):
                continue
            # Accumulate sequence lines
            lines.append(line)
    # Join lines into one continuous string of amino acids
    return "".join(lines)


def load_model(model_name):
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EsmForMaskedLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device


def load_rare_variants(csv_filename):
    """
    Loads a CSV with columns:
      rsId, Chromosome, Pos, Ref, Alt, AF
    Returns a list of dicts with typed fields.
    """
    variants = []
    with open(csv_filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["Pos"] = int(row["Pos"])  # 1-based position in the protein
            row["AF"] = float(row["AF"])  # allele frequency
            variants.append(row)
    return variants


def check_position(protein_seq, pos):
    aa_at_pos = -1
    if len(protein_seq) < pos:
        print(
            f"Error: Sequence only has {len(protein_seq)} amino acids; position {pos} is out of range."
        )
    else:
        aa_at_pos = protein_seq[pos - 1]
    print(f"Amino acid at position {pos} is: {aa_at_pos}")
    return aa_at_pos


if __name__ == "__main__":
    print(f"Curr.dir:{os.getcwd()}")

    # 1) Load rare variants and BRCA1 sequence
    filename = "./data/Homo_sapiens_ENSP00000350283_3_sequence.fa"
    protein_seq = load_protein_sequence(filename)
    print(f"Sequence length: {len(protein_seq)}")
    rare_variants = load_rare_variants("./data/BRCA1_rare_variants.csv")
    # check_position(rare_variants, 1296)

    # # 2) Load Model
    model_name = "facebook/esm2_t6_8M_UR50D"
    model, tokenizer, device = load_model(model_name)

    # 3) Run predictions
    results = run_predictions(protein_seq, rare_variants, model, tokenizer, device)

    # # 4) Print top 5
    # for r in results[:5]:
    #     print(r)
