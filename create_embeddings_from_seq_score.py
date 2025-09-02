#!/usr/bin/env python3

import pandas as pd
import torch
from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm


def create_fasta_from_seq_score(seq_score_path, fasta_path):
    df = pd.read_csv(seq_score_path)

    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            if pd.isna(row["mutations"]):
                label = "WT"
            else:
                label = row["mutations"].replace(":", "_")

            f.write(f">{label}\n")
            f.write(f"{row['sequence']}\n")


def generate_embeddings(fasta_path, output_dir):
    output_dir.mkdir(exist_ok=True, parents=True)

    cmd = [
        "python",
        "src/embeddings.py",
        "--fasta",
        str(fasta_path),
        "--output_dir",
        str(output_dir),
        "--truncation_seq_length",
        "238",
    ]

    subprocess.run(cmd, check=True)


def convert_mutation_format(mutation_str):
    if pd.isna(mutation_str):
        return "WT"

    parts = mutation_str.split(":")
    new_parts = []
    for part in parts:
        first_char = part[0]
        last_char = part[-1]
        number = int(part[1:-1])
        new_part = f"{first_char}{number - 2}{last_char}"
        new_parts.append(new_part)

    return "_".join(new_parts)


def process_embeddings_to_csv(seq_score_path, embeddings_dir, output_csv_path):
    seq_score_df = pd.read_csv(seq_score_path)

    mutation_to_data = {}
    wt_score = None

    for _, row in seq_score_df.iterrows():
        if pd.isna(row["mutations"]):
            wt_score = row["score"]
            mutation_to_data["WT"] = {"score": row["score"]}
        else:
            label = row["mutations"].replace(":", "_")
            mutation_to_data[label] = {"score": row["score"]}

    embedding_files = list(embeddings_dir.glob("*.pt"))
    results = []

    for emb_file in tqdm(embedding_files):
        data = torch.load(emb_file, map_location="cpu")
        label = data["label"]
        embedding = data["mean_representations"][47].numpy()

        score = mutation_to_data[label]["score"]

        if label == "WT":
            variant = "WT"
            var = "WT"
            num_mutations = 0
        else:
            variant = convert_mutation_format(label.replace("_", ":"))
            var = variant.replace(",", "_")
            num_mutations = var.count("_") + 1

        score_wt_norm = score - wt_score

        row = {
            "variant": variant,
            "score": score,
            "score_wt_norm": score_wt_norm,
            "var": var,
            "num_mutations": num_mutations,
        }

        for i in range(5120):
            row[str(i)] = embedding[i]

        results.append(row)

    df = pd.DataFrame(results)

    meta_cols = ["variant", "score", "score_wt_norm", "var", "num_mutations"]
    emb_cols = [str(i) for i in range(5120)]
    df = df[meta_cols + emb_cols]

    df.to_csv(output_csv_path, index=False)


def main():
    seq_score_path = "data/seq_and_score.csv"
    fasta_path = "temp_sequences.fasta"
    embeddings_dir = Path("temp_embeddings")
    output_csv_path = "data/esm2_15b_embeddings_and_meta_generated.csv"

    create_fasta_from_seq_score(seq_score_path, fasta_path)
    generate_embeddings(fasta_path, embeddings_dir)
    process_embeddings_to_csv(seq_score_path, embeddings_dir, output_csv_path)

    Path(fasta_path).unlink()
    shutil.rmtree(embeddings_dir)


if __name__ == "__main__":
    main()
