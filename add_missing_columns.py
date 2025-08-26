#!/usr/bin/env python3

import pandas as pd

def convert_mutation(mutation_str):
    """Convert A110D:N146D to A108D_N144D (subtract 2, : to _)"""
    parts = mutation_str.split(':')
    new_parts = []
    for part in parts:
        first_char = part[0]
        last_char = part[-1] 
        number = int(part[1:-1])
        new_part = f'{first_char}{number - 2}{last_char}'
        new_parts.append(new_part)
    return '_'.join(new_parts)

# Load files
embeddings = pd.read_csv('data/esm2_15b_embeddings_and_meta.csv')
seq_score = pd.read_csv('data/seq_and_score.csv')

# Create score mapping dictionary
score_mapping = {}
for _, row in seq_score.iterrows():
    if pd.notna(row['mutations']):
        converted_mutation = convert_mutation(row['mutations'])
        score_mapping[converted_mutation] = row['score']

# Add score column
embeddings['score'] = embeddings['variant'].map(score_mapping)

# Add score_wt_norm column
wt_score = 3.7192121319
embeddings['score_wt_norm'] = embeddings['score'] - wt_score

# Add var column
embeddings['var'] = embeddings['variant'].str.replace(',', '_')

# Add num_mutations column
embeddings['num_mutations'] = embeddings['var'].str.count('_') + 1

# Save updated file
embeddings.to_csv('data/esm2_15b_embeddings_and_meta_updated.csv', index=False)

print('Successfully added missing columns and saved to data/esm2_15b_embeddings_and_meta_updated.csv')