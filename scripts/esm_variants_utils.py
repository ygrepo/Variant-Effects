import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EsmForMaskedLM,
)
import os

AAorder = [
    "K",
    "R",
    "H",
    "E",
    "D",
    "N",
    "Q",
    "T",
    "S",
    "C",
    "G",
    "A",
    "V",
    "L",
    "I",
    "M",
    "P",
    "Y",
    "F",
    "W",
]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


##### INFERENCE
# def load_esm_model(model_name, device=0):
#     import torch

#     repr_layer = int(model_name.split("_")[1][1:])
#     model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
#     batch_converter = alphabet.get_batch_converter()
#     return model.eval().to(device), alphabet, batch_converter, repr_layer


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
            model_path, tokenizer_path = load_model_paths(model_name)
            print(f"Loading EVO model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.to(device)
            model.eval()

            print(f"Loading EVO tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            return model, tokenizer, device

    elif model_name == "facebook/esm1b_t33_650M_UR50S":
        if not load_flag:
            # Load ESM-1b model from torch.hub
            model, alphabet = torch.hub.load(
                "facebookresearch/esm", "esm1b_t33_650M_UR50S"
            )
            model.eval().to(device)

            # Save to local cache
            model_path, _ = load_model_paths(model_name)  # tokenizer not used
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Saving ESM-1b model to {model_path}")
            torch.save((model, alphabet), model_path)

            return model, alphabet, device
        else:
            # Load ESM-1b model from local cache
            model_path, _ = load_model_paths(model_name)  # tokenizer not used
            print(f"Loading ESM-1b model from {model_path}")
            model, alphabet = torch.load(model_path, map_location=device)
            return model.eval().to(device), alphabet, device

    elif model_name == "facebook/esm2_t6_8M_UR50D":
        # Load ESM-2 model
        # ========= ESM: Masked LM =========
        if not load_flag:
            # 1) Load from Hugging Face
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = EsmForMaskedLM.from_pretrained(model_name).to(device)
            model.eval()
            return model, tokenizer, device
        else:
            # 2) Load from local pickle
            model_path, tokenizer_path = load_model_paths(model_name)
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


def load_model_paths(model_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    pickle_dir = os.path.join(base_dir, "pickle")
    model_path = os.path.join(pickle_dir, model_name.replace("/", "_") + "_model")
    tokenizer_path = os.path.join(
        pickle_dir, model_name.replace("/", "_") + "_tokenizer"
    )
    return model_path, tokenizer_path


def get_wt_LLR(input_df, model, tokenizer, device="cuda", silent=False):
    """
    Compute Wild-Type Log-Likelihood Ratio (LLR) for protein sequences.
    Uses Hugging Face ESM model instead of Facebook's alphabet-based version.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Standard amino acid order
    AAorder = [
        "K",
        "R",
        "H",
        "E",
        "D",
        "N",
        "Q",
        "T",
        "S",
        "C",
        "G",
        "A",
        "V",
        "L",
        "I",
        "M",
        "P",
        "Y",
        "F",
        "W",
    ]

    LLRs = []
    input_df_ids = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), disable=silent):
        gname = row["id"]
        sequence = row["seq"]
        seq_length = len(sequence)

        # Ensure the sequence length matches the model's output
        if seq_length > 1022:
            print(f"Warning: {gname} sequence is too long ({seq_length}). Truncating!")
            sequence = sequence[:1022]  # Truncate to max ESM2 sequence length

        # Tokenization with attention_mask
        batch_tokens = tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1022,
        )
        batch_tokens = {k: v.to(device) for k, v in batch_tokens.items()}
        # Run ESM model
        with torch.no_grad():
            results_ = (
                torch.log_softmax(
                    model(
                        batch_tokens["input_ids"],
                        attention_mask=batch_tokens["attention_mask"],
                    )["logits"],
                    dim=-1,
                )
                .cpu()
                .numpy()
            )

        # Adjust the sequence length to match logits
        actual_seq_length = min(
            seq_length, results_.shape[1] - 2
        )  # Adjust for special tokens
        logit_data = results_[0, 1 : actual_seq_length + 1, :]  # Extract valid range

        # Extract WT log probabilities
        WTlogits = pd.DataFrame(
            logit_data,
            columns=tokenizer.get_vocab().keys(),
            index=list(sequence[:actual_seq_length]),  # Ensure correct index length
        ).T.loc[AAorder]

        WTlogits.columns = [f"{aa} {i+1}" for i, aa in enumerate(WTlogits.columns)]

        # Compute LLR
        wt_norm = np.diag(WTlogits.loc[[aa.split(" ")[0] for aa in WTlogits.columns]])
        LLR = WTlogits - wt_norm

        LLRs.append(LLR)
        input_df_ids.append(gname)

    return input_df_ids, LLRs


def get_logits(seq, model_type, model, tokenizer_or_alphabet, format=None, device=0):
    """
    Compute log-probabilities (logits) for a given sequence using either Hugging Face's ESM (ESM-2)
    or Facebook's ESM-1b model.

    Parameters:
        model_type: "esm2" (HuggingFace/ESM2) or "esm1" (Facebook ESM-1b)
        seq: protein sequence string
        model: loaded model
        tokenizer_or_alphabet: HuggingFace tokenizer or ESM alphabet
        format: if "pandas", returns a nicely formatted DataFrame
        device: torch.device or device index
    """
    # device = torch.device(device)
    # device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if model_type == "esm1":
        # Tokenize using ESM1's alphabet and batch_converter
        batch_converter = tokenizer_or_alphabet.get_batch_converter()
        data = [("sequence", seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            logits_tensor = model(batch_tokens, repr_layers=[], return_contacts=False)[
                "logits"
            ]
            logits = torch.log_softmax(logits_tensor, dim=-1).cpu().numpy()

        # Remove special tokens (<cls>, <eos>)
        logits = logits[0, 1:-1, :]  # Drop CLS and EOS
        aa_sequence = list(seq)

    elif model_type == "esm2":
        # Tokenize with Hugging Face
        tokens = tokenizer_or_alphabet(
            seq, return_tensors="pt", add_special_tokens=True
        ).to(device)

        with torch.no_grad():
            logits = torch.log_softmax(model(**tokens).logits, dim=-1).cpu().numpy()

        # Remove special tokens (CLS and EOS)
        logits = logits[0, 1:-1, :]

        aa_sequence = list(seq)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if format == "pandas":

        if model_type == "esm2":
            vocab = tokenizer_or_alphabet.get_vocab()
            id_to_token = {v: k for k, v in vocab.items()}
        else:  # esm1
            id_to_token = {
                i: tok for tok, i in tokenizer_or_alphabet.tok_to_idx.items()
            }

        aa_vocab = [aa for aa in "ACDEFGHIKLMNPQRSTVWY"]
        vocab_indices = [id_to_token[i] for i in range(logits.shape[-1])]

        df = pd.DataFrame(
            logits, columns=vocab_indices, index=range(len(aa_sequence))
        ).T.loc[
            aa_vocab
        ]  # Only keep standard AAs

        df.columns = [f"{res} {i+1}" for i, res in enumerate(aa_sequence)]
        return df

    return logits


def get_PLL(seq, model_type, model, tokenizer_or_alphabet, reduce=np.sum, device=None):
    """
    Compute the Protein Log-Likelihood (PLL) for a given sequence.
    """

    # Get log-probabilities
    s = get_logits(seq, model_type, model, tokenizer_or_alphabet, None, device)

    # Get token indices for ground truth amino acids
    if model_type == "esm1":
        idx = [tokenizer_or_alphabet.tok_to_idx[aa] for aa in seq]
    elif model_type == "esm2":
        idx = tokenizer_or_alphabet.encode(seq, add_special_tokens=False)
        # Get token indices manually using the ESM1 alphabet
    else:
        raise ValueError("Unsupported model type")

    return reduce(np.diag(s[:, idx]))


### MELTed CSV
def meltLLR(LLR, savedir=None):
    vars = LLR.melt(ignore_index=False)
    vars["variant"] = [
        "".join(i.split(" ")) + j for i, j in zip(vars["variable"], vars.index)
    ]
    vars["score"] = vars["value"]
    vars = vars.set_index("variant")
    vars["pos"] = [int(i[1:-1]) for i in vars.index]
    del vars["variable"], vars["value"]
    if savedir is not None:
        vars.to_csv(savedir + "var_scores.csv")
    return vars


def get_start_loss_LLR(seq, model, tokenizer, device):
    """
    Compute the LLR for start-loss mutations.
    - If the start codon is lost, look at downstream methionines (alternative starts).
    - Compute LLR at M1 and compare with alternative start sites.
    """
    # Compute Wild-Type LLR for the sequence
    seq_df = pd.DataFrame(
        [("_", "_", seq, len(seq))], columns=["id", "gene", "seq", "length"]
    )
    input_df_ids, LLRs = get_wt_LLR(
        seq_df, model, tokenizer, device=device, silent=True
    )

    if len(LLRs) == 0:
        return "N/A"  # No valid LLR computed

    llr_matrix = LLRs[0]  # Extract LLR matrix

    # Compute LLR at position 1 (Start Codon)
    try:
        start_loss_llr = llr_matrix.loc[
            "M", "M 1"
        ]  # Check if M1 exists in the LLR matrix
    except KeyError:
        print("Warning: M1 not found in LLR matrix. Check tokenization.")
        return "N/A"

    # Find alternative start sites (Methionines)
    alternative_sites = [col for col in llr_matrix.columns if col.startswith("M ")]
    alternative_llrs = [
        llr_matrix.loc["M", col]
        for col in alternative_sites
        if int(col.split(" ")[1]) > 1
    ]

    # If alternative start sites exist, return the lowest LLR
    if alternative_llrs:
        return min(
            start_loss_llr, min(alternative_llrs)
        )  # Use the most likely alternative start

    return start_loss_llr  # If no alternative, return M1 LLR


def compute_delins_llr(
    model_type,
    model,
    tokenizer_or_alphabet,
    wt_seq,
    mut_seq,
    position,
    alt_aa,
    device,
):
    """
    Compute the Log-Likelihood Ratio (LLR) for a Delins mutation.

    Parameters:
        model_type:  "esm1" (Facebook ESM-1b) or "esm2" (Hugging Face)
        model: ESM model (Hugging Face or ESM1)
        tokenizer_or_alphabet: either HuggingFace tokenizer or ESM1 Alphabet
        wt_seq: wild-type sequence (string)
        mut_seq: mutant sequence (string)
        position: 0-based index (residue position)
        alt_aa: mutated residue (e.g. 'V')
        device: torch.device
    """
    if model_type == "esm2":
        # Tokenize both sequences
        wt_tokens = tokenizer_or_alphabet(
            wt_seq, return_tensors="pt", padding=True, truncation=True
        )
        mut_tokens = tokenizer_or_alphabet(
            mut_seq, return_tensors="pt", padding=True, truncation=True
        )

        wt_tokens = wt_tokens["input_ids"].to(device)
        mut_tokens = mut_tokens["input_ids"].to(device)

        with torch.no_grad():
            wt_logits = (
                torch.log_softmax(model(wt_tokens)["logits"], dim=-1).cpu().numpy()
            )
            mut_logits = (
                torch.log_softmax(model(mut_tokens)["logits"], dim=-1).cpu().numpy()
            )

        # Token position is 0-based directly
        wt_ll = wt_logits[0, position, :]
        mut_ll = mut_logits[0, position, :]

        target_token_id = tokenizer_or_alphabet.encode(
            alt_aa, add_special_tokens=False
        )[0]
    elif model_type == "esm1":
        # First truncate sequence (if needed) to avoid ESM1's 1024-token limit
        wt_seq, mut_seq, position = center_truncate(wt_seq, mut_seq, position)

        # Tokenize using ESM1's batch converter
        batch_converter = tokenizer_or_alphabet.get_batch_converter()
        data = [("WT", wt_seq), ("MUT", mut_seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            logits = model(batch_tokens, repr_layers=[], return_contacts=False)[
                "logits"
            ]
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        # Adjust for <cls> token at index 0
        wt_ll = log_probs[0, position + 1, :]
        mut_ll = log_probs[1, position + 1, :]

        target_token_id = tokenizer_or_alphabet.tok_to_idx[alt_aa]

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Compute log-likelihood ratio
    llr = mut_ll[target_token_id] - wt_ll[target_token_id]

    return llr


def get_local_PLL(seq, model_type, model, tokenizer, device=0):
    """
    Compute local PLL values for each position in the sequence.
    This function obtains the logits from the model, then extracts the PLL value
    (the log-likelihood of the actual residue) for each position.
    """
    # Get logits from the model for the entire sequence.
    s = get_logits(
        seq, model_type, model=model, tokenizer=tokenizer, format=None, device=device
    )
    # Encode the sequence (without special tokens) to obtain token indices.
    idx = tokenizer.encode(seq, add_special_tokens=False)
    # Extract the PLL for each position as the logit corresponding to the actual residue.
    # (This assumes that a higher logit corresponds to a higher likelihood.)
    local_pll = np.array([s[i, idx[i]] for i in range(len(idx))])
    return local_pll


def center_truncate(wt_seq, mut_seq, position, max_len=1022):
    half = max_len // 2
    start = max(0, position - half)
    end = min(len(wt_seq), position + half)

    new_wt = wt_seq[start:end]
    new_mut = mut_seq[start:end]
    new_pos = position - start

    return new_wt, new_mut, new_pos


##################### TILING utils ###########################


def chop(L, min_overlap=511, max_len=1022):
    return L[max_len - min_overlap : -max_len + min_overlap]


def intervals(L, min_overlap=511, max_len=1022, parts=None):
    if parts is None:
        parts = []
    # print('L:',len(L))
    # print(len(parts))
    if len(L) <= max_len:
        if parts[-2][-1] - parts[-1][0] < min_overlap:
            # print('DIFF:',parts[-2][-1]-parts[-1][0])
            return parts + [
                np.arange(
                    L[int(len(L) / 2)] - int(max_len / 2),
                    L[int(len(L) / 2)] + int(max_len / 2),
                )
            ]
        else:
            return parts
    else:
        parts += [L[:max_len], L[-max_len:]]
        L = chop(L, min_overlap, max_len)
        return intervals(L, min_overlap, max_len, parts=parts)


def get_intervals_and_weights(seq_len, min_overlap=511, max_len=1022, s=16):
    ints = intervals(np.arange(seq_len), min_overlap=min_overlap, max_len=max_len)
    ## sort intervals
    ints = [ints[i] for i in np.argsort([i[0] for i in ints])]

    a = int(np.round(min_overlap / 2))
    t = np.arange(max_len)

    f = np.ones(max_len)
    f[:a] = 1 / (1 + np.exp(-(t[:a] - a / 2) / s))
    f[max_len - a :] = 1 / (1 + np.exp((t[:a] - a / 2) / s))

    f0 = np.ones(max_len)
    f0[max_len - a :] = 1 / (1 + np.exp((t[:a] - a / 2) / s))

    fn = np.ones(max_len)
    fn[:a] = 1 / (1 + np.exp(-(t[:a] - a / 2) / s))

    filt = [f0] + [f for i in ints[1:-1]] + [fn]
    M = np.zeros((len(ints), seq_len))
    for k, i in enumerate(ints):
        M[k, i] = filt[k]
    M_norm = M / M.sum(0)
    return (ints, M, M_norm)


## PLLR score for indels
def get_PLLR(
    wt_seq,
    mut_seq,
    start_pos,
    model_type,
    model,
    tokenizer,  # Replace alphabet with tokenizer
    weighted=False,
    device=0,
):
    """
    Compute PLLR by subtracting WT PLL score from Mutant PLL score.
    """
    fn = np.sum if not weighted else np.mean

    if max(len(wt_seq), len(mut_seq)) <= 1022:
        return get_PLL(
            mut_seq,
            model_type,
            model,
            tokenizer,  # Pass tokenizer to the function
            reduce=fn,
            device=device,
        ) - get_PLL(
            wt_seq,
            model_type,
            model,
            tokenizer,  # Pass tokenizer to the function
            fn,
            device,
        )
    else:
        wt_seq, mut_seq, start_pos = crop_indel(wt_seq, mut_seq, start_pos)
        return get_PLL(
            mut_seq,
            model_type,
            model,
            tokenizer,
            fn,
            device,
        ) - get_PLL(
            wt_seq,
            model_type,
            model,
            tokenizer,
            fn,
            device,
        )


def crop_indel(ref_seq, alt_seq, ref_start):
    max_len = 1022  # Maximum length allowed by ESM model
    left_pos = ref_start - 1  # Convert 1-based index to 0-based

    # If sequence is already short enough, return unchanged
    if len(ref_seq) <= max_len and len(alt_seq) <= max_len:
        return ref_seq, alt_seq, ref_start  # Keep mutation position unchanged

    # Center the mutation in the cropped sequence
    start_pos = max(0, left_pos - max_len // 2)
    end_pos1 = min(start_pos + max_len, len(ref_seq))  # Crop for WT
    end_pos2 = min(start_pos + max_len, len(alt_seq))  # Crop for Mutant

    # Adjust cropping to ensure mutation remains visible
    if left_pos < start_pos:
        start_pos = max(0, left_pos - 50)  # Shift left to keep mutation
        end_pos1 = min(start_pos + max_len, len(ref_seq))
        end_pos2 = min(start_pos + max_len, len(alt_seq))

    adj_pos = ref_start - start_pos  # Fix mutation position adjustment

    return ref_seq[start_pos:end_pos1], alt_seq[start_pos:end_pos2], adj_pos


## stop gain variant score
def get_minLLR(seq, stop_pos, model, tokenizer, device=0):
    """
    Compute the minimum LLR score after a given stop position.

    - Used for stop-loss variants.
    - Extracts the log-likelihood ratio (LLR) matrix for all positions.
    - Returns the **minimum LLR value after stop_pos**.
    """
    # Convert sequence into DataFrame (consistent with `get_wt_LLR` input format)
    seq_df = pd.DataFrame(
        [("_", "_", seq, len(seq))], columns=["id", "gene", "seq", "length"]
    )

    # Compute LLR matrix for the wild-type sequence
    input_df_ids, LLRs = get_wt_LLR(
        seq_df, model, tokenizer, device=device, silent=True
    )

    # Ensure we have valid LLR values
    if len(LLRs) == 0:
        print("Warning: No LLR values computed!")
        return "N/A"

    # Extract LLR matrix
    llr_matrix = LLRs[0]

    # Ensure stop_pos is within bounds
    if stop_pos >= llr_matrix.shape[1]:
        print(
            f"Warning: stop_pos {stop_pos} is beyond sequence length {llr_matrix.shape[1]}"
        )
        return "N/A"

    # Extract values after stop_pos and find the minimum
    return np.min(llr_matrix.values[:, stop_pos:])


# ############### EXAMPLE ##################
if __name__ == "__main__":
    # ## Load model
    model_name = "facebook/esm1b_t33_650M_UR50S"
    model, tokenizer, device = load_model(model_name=model_name, load_flag=False)
    print(f"Model: {model_name} loaded successfully!")
    # model,alphabet,batch_converter,repr_layer = load_esm_model(model_name='esm1b_t33_650M_UR50S',device='cuda')
    ## Create a toy dataset
    # df_in = pd.DataFrame(
    #     [
    #         ("P1", "gene1", "FISHWISHFQRCHIPSTHATARECRISP", 28),
    #         ("P2", "gene2", "RAGEAGAINSTTHEMACHINE", 21),
    #         ("P3", "gene3", "SHIPSSAILASFISHSWIM", 19),
    #         ("P4", "gene4", "A" * 1948, 1948),
    #     ],
    #     columns=["id", "gene", "seq", "length"],
    # )
    # ## Get LLRs
    # ids, LLRs = get_wt_LLR(
    #     df_in, model=model, tokenizer=tokenizer, device=device, silent=False
    # )
    # for i, LLR in zip(ids, LLRs):
    #     print(i, LLR.shape)
    # sequence = "MADEEKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGNAV"
    # logits = get_logits(sequence, model, tokenizer, format="pandas", device="cpu")
    # print(logits)

    # ## Get PLL
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(get_PLL(df_in.seq.values[0], model, tokenizer, device=device))
    # # indel: 14_IPS_delins_EESE (FISHWISHFQRCHIPSTHATARECRISP --> FISHWISHFQRCHEESETHATARECRISP)
    # print(
    #     get_PLLR(
    #         "FISHWISHFQRCHIPSTHATARECRISP",
    #         "FISHWISHFQRCHEESETHATARECRISP",
    #         14,
    #         model,
    #         tokenizer,  # Replace alphabet with tokenizer
    #         weighted=False,
    #         device=device,
    #     )
    # )
    # ## stop at position 17
    # print(get_minLLR(df_in.seq.values[0], 17, model, tokenizer, device=device))
    # ref_seq = "MKVLWAALLVTFLAGCQAKVE"  # 21 amino acids (WT)
    # alt_seq = "MKVLWAALLVTFLAGCQAKVEE"  # 22 amino acids (Mutant with insertion)
    # ref_start = 10  # Indel at position 10
    # ref_cropped, alt_cropped, adj_pos = crop_indel(ref_seq, alt_seq, ref_start)
    # print(ref_cropped)  # "MKVLWAALLVTF" (1022-length)
    # print(alt_cropped)  # "MKVLWAALLVTF" (1022-length)
    # print(adj_pos)  # Adjusted position relative to cropped sequence
