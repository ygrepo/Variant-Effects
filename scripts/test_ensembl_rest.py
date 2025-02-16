import requests


def convert_grch37_to_grch38(chrom, pos):
    """
    Converts GRCh37 coordinates to GRCh38 using Ensembl's assembly converter API.
    """
    url = f"https://rest.ensembl.org/map/human/GRCh37/{chrom}:{pos}..{pos}/GRCh38?"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if "mappings" in data and len(data["mappings"]) > 0:
            return data["mappings"][0]["mapped"]["start"]  # Return converted position
    return None  # If conversion fails


def get_transcript_id(chrom, pos, ref, alt):
    """
    Queries Ensembl VEP API to retrieve the transcript ID (ENST) for a given genomic variant.
    Converts GRCh37 to GRCh38 before querying if necessary.
    """
    # Convert position to GRCh38
    new_pos = convert_grch37_to_grch38(chrom, pos)

    if new_pos is None:
        print(f"Failed to convert position: {chrom}:{pos}")
        return None  # Skip if conversion fails

    # Query VEP API using GRCh38 position
    url = f"https://rest.ensembl.org/vep/human/region/{chrom}:{new_pos}/{ref}/{alt}?"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        results = response.json()
        if results and "transcript_consequences" in results[0]:
            for transcript in results[0]["transcript_consequences"]:
                if "transcript_id" in transcript:
                    return transcript["transcript_id"]  # Return first found ENST ID

    return None  # If no transcript is found


# Example usage
chrom = "1"
pos = "17018883"  # GRCh37 position
ref = "A"
alt = "C"

transcript_id = get_transcript_id(chrom, pos, ref, alt)
print(f"Transcript ID: {transcript_id}")
