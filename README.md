# GFP Variant Design

CNN ensemble model for protein activity prediction using ESM-2 embeddings.

## Installation

### Prerequisites
- Python 3.12.1
- pip

**Note:** All versions of Python and packages are pinned to the exact versions used during analysis for reproducibility.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Arcadia-Science/2025-GFP-variant-design.git
cd 2025-GFP-variant-design
```

2. Install dependencies:
```bash
pip install -e .
```

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

## Usage

Run notebooks in order: `01_train_model.ipynb` → `02_generate_sequences.ipynb` → `03_create_figures.ipynb`

### Generate Embeddings from Original Dataset

Converts `seq_and_score.csv` to full embeddings dataset with ESM-2 embeddings and metadata columns.

Some systems need certificates for embedding generation:

```bash
apt update && apt install -y ca-certificates && update-ca-certificates
```

Usage:
```bash
python create_embeddings_from_seq_score.py
```

**Performance Notes:**
- Original embeddings (~50k sequences) were generated on an H100 GPU in ~9 hours
- Uses GPU-to-CPU memory offloading, allowing the ESM-2 15B model to run on GPUs with limited VRAM by storing model parameters in CPU memory and transferring them to GPU as needed during computation
- Can run on smaller GPUs (e.g., A10G ~35 hours) but will take significantly longer
- Requires significant computational resources and CUDA-compatible GPU