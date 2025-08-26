# GFP Variant Design

CNN ensemble model for protein activity prediction using ESM-2 embeddings.

## Notebooks

### 01_train_model.ipynb
Trains a CNN ensemble model on protein sequence embeddings and activity scores.

### 02_generate_sequences.ipynb  
Generates protein sequence variants and predicts their activities using the trained model.

### 03_create_figures.ipynb
Creates visualizations from the results.

## Scripts

### create_embeddings_from_seq_score.py
Converts `seq_and_score.csv` to embeddings file:
1. Creates FASTA from sequences
2. Generates ESM-2 embeddings 
3. Adds metadata columns

### add_missing_columns.py
Adds missing columns to existing embeddings file from `seq_and_score.csv`.

## System Requirements

Some systems need certificates for embedding generation:

```bash
apt update && apt install -y ca-certificates && update-ca-certificates
```

## Usage

Run notebooks in order: `01_train_model.ipynb` → `02_generate_sequences.ipynb` → `03_create_figures.ipynb`

To generate embeddings from scratch:
```bash
python create_embeddings_from_seq_score.py
```