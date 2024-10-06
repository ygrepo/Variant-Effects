from transformers import AutoModelForMaskedLM, AutoTokenizer
import os
import requests
import json
import pickle

# Downloads and installs the model, tokenizer, and DNA data. Required as the compute nodes do not have network access.


def setup_model_and_tokenizer(model_name, pickle_dir):
    # Create the model and tokenizer directories based on their names
    model_dir = os.path.join(pickle_dir, model_name.replace("/", "_") + "_model")
    tokenizer_dir = os.path.join(
        pickle_dir, model_name.replace("/", "_") + "_tokenizer"
    )

    # Ensure the directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    print(f"The working directory is {os.getcwd()}")

    print("Downloading model and tokenizer...")
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Save the model and tokenizer in the specified directories
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    print(f"Model saved to {model_dir}")
    print(f"Tokenizer saved to {tokenizer_dir}")


def fetch_and_save_dna_sequence():
    # Prompt the user for the DNA sequence name
    sequence_name = input("Enter the DNA sequence ID (e.g., ENSG00000012048): ")

    # Make a request to the Ensembl REST API to retrieve the DNA sequence
    url = f"https://rest.ensembl.org/sequence/id/{sequence_name}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        dna_sequence = data["seq"]
        sequence_length = len(dna_sequence)

        # Check if the sequence exceeds the length limit
        if sequence_length > 131000:
            print(
                f"The sequence length is {sequence_length}, which exceeds the limit of 131,000 bases."
            )
            start = int(input("Enter the start position: "))
            end = int(input("Enter the end position: "))
            dna_sequence = dna_sequence[start - 1 : end]
            data["seq"] = dna_sequence  # Update the sequence in data

        # Save the data to a JSON file
        json_file_path = "dna_data.json"
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file)

        print(f"Data saved to {json_file_path}")
    else:
        print("Failed to retrieve the DNA sequence from Ensembl.")
        print("Status Code:", response.status_code)


if __name__ == "__main__":
    MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    pickle_dir = "../pickle"

    setup_model_and_tokenizer(MODEL_NAME, pickle_dir)
    # fetch_and_save_dna_sequence()
