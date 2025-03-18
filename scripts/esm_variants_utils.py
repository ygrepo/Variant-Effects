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


def get_wt_LLR(input_df, model, tokenizer, device="cuda", silent=False):
    """
    Compute Wild-Type Log-Likelihood Ratio (LLR) for protein sequences.
    Supports Hugging Face's ESM model instead of Facebook's alphabet.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

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

    genes = input_df["id"].values
    LLRs = []
    input_df_ids = []

    for gname in tqdm(genes, disable=silent):
        seq_length = input_df[input_df["id"] == gname]["length"].values[0]
        sequence = input_df[input_df["id"] == gname]["seq"].values[0]

        if seq_length <= 1022:
            # **Tokenize using Hugging Face's EsmTokenizer**
            batch_tokens = tokenizer(
                sequence, return_tensors="pt", padding=True, truncation=True
            )
            batch_tokens = batch_tokens["input_ids"].to(device)

            # **Run ESM model**
            with torch.no_grad():
                results_ = (
                    torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
                    .cpu()
                    .numpy()
                )

            # **Extract WT log probabilities**
            WTlogits = pd.DataFrame(
                results_[0, 1:-1, :],  # Remove special tokens
                columns=tokenizer.get_vocab().keys(),  # Use Hugging Face tokenizer vocab
                index=list(sequence),
            ).T.loc[AAorder]

            WTlogits.columns = [
                j.split(".")[0] + " " + str(i + 1)
                for i, j in enumerate(WTlogits.columns)
            ]

            wt_norm = np.diag(WTlogits.loc[[i.split(" ")[0] for i in WTlogits.columns]])
            LLR = WTlogits - wt_norm

            LLRs.append(LLR)
            input_df_ids.append(gname)

        else:
            ### **Tiling for Long Sequences**
            long_seq = sequence
            ints, M, M_norm = get_intervals_and_weights(
                len(long_seq), min_overlap=512, max_len=1022, s=20
            )

            dt = ["".join(np.array(list(long_seq))[idx]) for idx in ints]
            logit_parts = []

            for dt_ in chunks(dt, 20):
                batch_tokens = tokenizer(
                    dt_, return_tensors="pt", padding=True, truncation=True
                )
                batch_tokens = batch_tokens["input_ids"].to(device)

                with torch.no_grad():
                    results_ = (
                        torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
                        .cpu()
                        .numpy()
                    )

                for i in range(results_.shape[0]):
                    logit_parts.append(results_[i, 1:-1, :])

            # **Merge tiled logits**
            logits_full = np.zeros((len(long_seq), len(AAorder)))
            for i in range(len(ints)):
                logit = np.zeros((len(long_seq), len(AAorder)))
                logit[ints[i]] = logit_parts[i]
                logit = np.multiply(logit.T, M_norm[i, :]).T
                logits_full += logit

            WTlogits = pd.DataFrame(
                logits_full,
                columns=tokenizer.get_vocab().keys(),
                index=list(sequence),
            ).T.loc[AAorder]

            WTlogits.columns = [
                j.split(".")[0] + " " + str(i + 1)
                for i, j in enumerate(WTlogits.columns)
            ]

            wt_norm = np.diag(WTlogits.loc[[i.split(" ")[0] for i in WTlogits.columns]])
            LLR = WTlogits - wt_norm

            LLRs.append(LLR)
            input_df_ids.append(gname)

    return input_df_ids, LLRs


def get_logits(seq, model, tokenizer, format=None, device=0):
    """
    Compute log-probabilities (logits) for a given sequence using Hugging Face's ESM model.
    """
    # Tokenize input sequence
    tokens = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)

    # Get log probabilities from model
    with torch.no_grad():
        logits = torch.log_softmax(model(**tokens).logits, dim=-1).cpu().numpy()

    # Remove special tokens from logits (first & last)
    logits = logits[0, 1:-1, :]

    if format == "pandas":
        # Create a DataFrame with log-probabilities for each amino acid
        WTlogits = pd.DataFrame(
            logits,
            columns=tokenizer.get_vocab().keys(),  # ✅ Fix: Use Hugging Face tokenizer vocabulary
            index=list(seq),
        ).T.loc[
            AAorder
        ]  # Select only standard amino acids

        # Rename columns to include residue position
        WTlogits.columns = [
            j.split(".")[0] + " " + str(i + 1) for i, j in enumerate(WTlogits.columns)
        ]
        return WTlogits
    else:
        return logits  # Return raw logits matrix


def get_PLL(seq, model, tokenizer, reduce=np.sum, device=0):
    """
    Compute the Protein Log-Likelihood (PLL) for a given sequence.
    """
    s = get_logits(
        seq,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    idx = tokenizer.encode(
        seq, add_special_tokens=False
    )  # ✅ Fix: Replace `alphabet.tok_to_idx[]`
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
            model=model,
            tokenizer=tokenizer,  # Pass tokenizer to the function
            reduce=fn,
            device=device,
        ) - get_PLL(
            wt_seq,
            model=model,
            tokenizer=tokenizer,  # Pass tokenizer to the function
            reduce=fn,
            device=device,
        )
    else:
        wt_seq, mut_seq, start_pos = crop_indel(wt_seq, mut_seq, start_pos)
        return get_PLL(
            mut_seq,
            model=model,
            tokenizer=tokenizer,
            reduce=fn,
            device=device,
        ) - get_PLL(
            wt_seq,
            model=model,
            tokenizer=tokenizer,
            reduce=fn,
            device=device,
        )


def crop_indel(ref_seq, alt_seq, ref_start):
    # Start pos: 1-indexed start position of variant
    left_pos = ref_start - 1
    offset = len(ref_seq) - len(alt_seq)
    start_pos = int(left_pos - 1022 / 2)
    end_pos1 = int(left_pos + 1022 / 2) - min(start_pos, 0) + min(offset, 0)
    end_pos2 = int(left_pos + 1022 / 2) - min(start_pos, 0) - max(offset, 0)
    if start_pos < 0:
        start_pos = 0  # Make sure the start position is not negative
    if end_pos1 > len(ref_seq):
        end_pos1 = len(
            ref_seq
        )  # Make sure the end positions are not beyond the end of the sequence
    if end_pos2 > len(alt_seq):
        end_pos2 = len(alt_seq)
    if (
        start_pos > 0 and max(end_pos2, end_pos1) - start_pos < 1022
    ):  ## extend to the left if there's space
        start_pos = max(0, max(end_pos2, end_pos1) - 1022)

    return (
        ref_seq[start_pos:end_pos1],
        alt_seq[start_pos:end_pos2],
        start_pos - ref_start,
    )


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


# ############### EXAMLE ##################
# ## Load model
# model,alphabet,batch_converter,repr_layer = load_esm_model(model_name='esm1b_t33_650M_UR50S',device='cuda')
# ## Create a toy dataset
# df_in = pd.DataFrame([('P1','gene1','FISHWISHFQRCHIPSTHATARECRISP',28),
#                       ('P2','gene2','RAGEAGAINSTTHEMACHINE',21),
#                       ('P3','gene3','SHIPSSAILASFISHSWIM',19),
#                       ('P4','gene4','A'*1948,1948)], columns = ['id','gene','seq','length'])
# ## Get LLRs
# ids,LLRs = get_wt_LLR(df_in)
# for i,LLR in zip(ids,LLRs):
#   print(i,LLR.shape)
# ## Get PLL
# print(get_PLL(df_in.seq.values[0]))
# ## indel: 14_IPS_delins_EESE (FISHWISHFQRCHIPSTHATARECRISP --> FISHWISHFQRCHEESETHATARECRISP)
# get_PLLR('FISHWISHFQRCHIPSTHATARECRISP','FISHWISHFQRCHEESETHATARECRISP',14)
# ## stop at position 17
# get_minLLR(df_in.seq.values[0],17)
