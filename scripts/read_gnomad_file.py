import csv
import os

AF_THRESHOLD = 6.24e-07  # example threshold (0.1%)


def read_gnomad_file(filename):
    print(f"Processing file: {filename}")
    rare_variants = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                af = float(row["Allele Frequency"])
            except ValueError:
                continue  # skip rows that don't have a valid AF

            if af <= AF_THRESHOLD:
                # Keep this variant
                chromosome = row["Chromosome"]
                position = int(row["Position"])
                rsId = row["rsIDs"]
                ref = row["Reference"]
                alt = row["Alternate"]
                hgvsp = row["HGVS Consequence"]  # e.g. "p.Leu42Pro"
                pCons = row["Protein Consequence"]  # e.g. "p.Leu42Pro"
                rare_variants.append(
                    {
                        "Chromosome": chromosome,
                        "Pos": position,
                        "rsId": rsId,
                        "Ref": ref,
                        "Alt": alt,
                        "HGVSp": hgvsp,
                        "pCons": pCons,
                        "AF": af,
                    }
                )

    print(f"Number of variants below AF={AF_THRESHOLD}: {len(rare_variants)}")
    print(f"Processing complete.")
    return rare_variants


if __name__ == "__main__":
    print(f"Curr.dir:{os.getcwd()}")
    rare_variants = read_gnomad_file(
        "./data/gnomAD_v4.1.0_ENSG00000012048_2025_01_18_11_16_59.csv"
    )
    # Save the rare variants to a new CSV file
    with open("./data/BRCA1_rare_variants.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rsId",
            "Chromosome",
            "Pos",
            "Ref",
            "Alt",
            "HGVSp",
            "pCons",
            "AF",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rare_variants)
