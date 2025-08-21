#!/usr/bin/env python3
"""
Modal Training Script for CNN Ensemble Multi-Run Training
Trains 10 CNN models with different random seeds to analyze ranking stability
"""

import modal
import pickle
import json
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path

# Modal app setup
app = modal.App("cnn-ensemble-multi-training")

def create_shared_image():
    """Create shared image with CNN ensemble source code."""
    return (
        modal.Image.debian_slim()
        .pip_install([
            "torch", 
            "torchvision", 
            "pandas", 
            "numpy", 
            "scikit-learn", 
            "tqdm"
        ])
        .add_local_dir("src", "/pkg/src")
    )

# Create shared components
image = create_shared_image()
volume = modal.Volume.from_name("cnn-training-data")
results_volume = modal.Volume.from_name("cnn-training-results", create_if_missing=True)

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

@dataclass
class TrainingResults:
    """Results from a single training run."""
    run_id: int
    seed: int
    final_r2: float
    train_losses: List[float]
    val_losses: List[float]
    val_r2_history: List[float]
    model_state_dict: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

def save_training_results(results: TrainingResults, job_name: str) -> str:
    """Save training results to files."""
    output_dir = f"/mnt/results/{job_name}/run_{results.run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    import torch
    torch.save(results.model_state_dict, f"{output_dir}/model.pth")
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(len(results.train_losses)),
        'train_loss': results.train_losses,
        'val_loss': results.val_losses,
        'val_r2': results.val_r2_history
    })
    history_df.to_csv(f"{output_dir}/training_history.csv", index=False)
    
    # Save full results
    with open(f"{output_dir}/results.pkl", "wb") as f:
        pickle.dump(results.to_dict(), f)
    
    # Save summary metrics
    metrics = {
        'run_id': results.run_id,
        'seed': results.seed,
        'final_r2': results.final_r2,
        'final_train_loss': results.train_losses[-1],
        'final_val_loss': results.val_losses[-1],
        'best_val_r2': max(results.val_r2_history)
    }
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return output_dir

def load_wild_type_sequence(csv_path: str = "/mnt/data/seq_and_score.csv") -> str:
    """Load the wild-type sequence from the CSV file.
    
    Args:
        csv_path: Path to the CSV file containing sequences and scores
        
    Returns:
        The wild-type sequence (sequence with no mutations)
    """
    df = pd.read_csv(csv_path)
    # Find the row with empty mutations (wild-type)
    wt_row = df[df['mutations'].isna() | (df['mutations'] == '')]
    if wt_row.empty:
        raise ValueError("No wild-type sequence found in the CSV file (no row with empty mutations)")
    return wt_row.iloc[0]['sequence']

@app.function(
    image=image,
    volumes={
        "/mnt/data": volume,
        "/mnt/results": results_volume
    },
    gpu="A10G",
    timeout=14400,
)
def train_single_run(run_id: int, seed: int, job_name: str):
    """Train a single CNN ensemble model with specified seed."""
    import sys
    sys.path.append("/pkg")
    
    import torch
    import numpy as np
    from src import models, training, data
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run {run_id} (seed {seed}): Using device {device}")
    
    # Load data
    df = pd.read_csv("/mnt/data/esm2_15b_embeddings_and_meta.csv")
    
    # Create data holder and loaders
    data_holder = data.ESMDataHolder(df)
    train_loader, val_loader = data_holder.train_val_split()
    
    # Create model and config
    model = models.Ensemble().to(device)
    config = TrainingConfig()
    
    # Use centralized training function
    training_result = training.train_variant_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        updates=True
    )
    
    # Create results object
    results = TrainingResults(
        run_id=run_id,
        seed=seed,
        final_r2=training_result['final_r2'],
        train_losses=training_result['train_losses'],
        val_losses=training_result['val_losses'],
        val_r2_history=training_result['val_r2_history'],
        model_state_dict=training_result['model'].cpu().state_dict()
    )
    
    # Save results
    output_dir = save_training_results(results, job_name)
    print(f"Run {run_id} complete. Results saved to {output_dir}")
    print(f"Final R²: {results.final_r2:.4f}")
    
    return results.to_dict()

@app.function(
    image=image,
    volumes={
        "/mnt/data": volume,
        "/mnt/results": results_volume
    },
    gpu="A10G",
    timeout=14400,
)
def generate_predictions_for_run(run_id: int, job_name: str):
    """Generate predictions using a trained model."""
    import sys
    sys.path.append("/pkg")
    
    import torch
    import numpy as np
    from src import models
    from pathlib import Path
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained model
    model_path = f"/mnt/results/{job_name}/run_{run_id}/model.pth"
    model = models.Ensemble().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load embedding files
    embedding_dir = Path("/mnt/data/embeddings")
    embedding_files = list(embedding_dir.glob("*.pt"))
    
    # Load wild-type sequence from CSV file
    wt_seq = load_wild_type_sequence()
    
    labels = []
    embeddings = []
    sequences = []
    
    print(f"Processing {len(embedding_files)} embedding files for run {run_id}...")
    
    for emb_file in tqdm(embedding_files):
        data = torch.load(emb_file)
        label = data["label"]
        embedding = data["mean_representations"][47]
        
        # Convert label to actual sequence
        seq = list(wt_seq)
        for mutation in label.split(":"):
            pos = int(mutation[:-1])
            new_aa = mutation[-1]
            seq[pos] = new_aa
        sequence = "".join(seq)
        
        labels.append(label)
        embeddings.append(embedding)
        sequences.append(sequence)
    
    # Generate predictions
    embeddings_tensor = torch.from_numpy(np.array(embeddings)).float().to(device)
    
    with torch.no_grad():
        predictions = model(embeddings_tensor).cpu().detach().numpy().flatten()
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "label": labels,
        "sequence": sequences,
        "predicted_score": predictions,
        "run_id": run_id
    })
    
    # Sort by predicted score (highest first)
    results_df = results_df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    results_df['rank'] = range(1, len(results_df) + 1)
    
    # Save predictions
    output_dir = f"/mnt/results/{job_name}/run_{run_id}"
    results_df.to_csv(f"{output_dir}/predictions.csv", index=False)
    
    print(f"Run {run_id} predictions complete. Score range: {results_df['predicted_score'].min():.3f} to {results_df['predicted_score'].max():.3f}")
    
    return results_df.to_dict('records')

@app.function(
    image=image,
    volumes={"/mnt/results": results_volume},
    timeout=600,
)
def analyze_ranking_stability(job_name: str, num_runs: int = 10):
    """Analyze ranking stability across different training runs."""
    all_predictions = []
    all_metrics = []
    
    # Collect all predictions and metrics
    for run_id in range(1, num_runs + 1):
        # Load predictions
        pred_file = f"/mnt/results/{job_name}/run_{run_id}/predictions.csv"
        df = pd.read_csv(pred_file)
        all_predictions.append(df)
        
        # Load metrics
        metrics_file = f"/mnt/results/{job_name}/run_{run_id}/metrics.json"
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
    
    # Create ranking comparison
    ranking_comparison = []
    
    for _, row in all_predictions[0].iterrows():
        label = row['label']
        sequence = row['sequence']
        
        ranks = []
        scores = []
        
        for df in all_predictions:
            seq_data = df[df['label'] == label]
            ranks.append(seq_data.iloc[0]['rank'])
            scores.append(seq_data.iloc[0]['predicted_score'])
        
        ranking_comparison.append({
            'label': label,
            'sequence': sequence,
            'mean_rank': np.mean(ranks),
            'std_rank': np.std(ranks),
            'min_rank': np.min(ranks),
            'max_rank': np.max(ranks),
            'rank_range': np.max(ranks) - np.min(ranks),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'ranks_across_runs': ranks,
            'scores_across_runs': scores
        })
    
    ranking_df = pd.DataFrame(ranking_comparison)
    ranking_df = ranking_df.sort_values('mean_rank')
    
    # Calculate overall statistics
    training_stats = pd.DataFrame(all_metrics)
    
    summary_stats = {
        'num_runs': len(all_metrics),
        'mean_final_r2': float(training_stats['final_r2'].mean()),
        'std_final_r2': float(training_stats['final_r2'].std()),
        'mean_final_train_loss': float(training_stats['final_train_loss'].mean()),
        'mean_final_val_loss': float(training_stats['final_val_loss'].mean()),
        'ranking_stability': {
            'mean_rank_std': float(ranking_df['std_rank'].mean()),
            'sequences_with_stable_ranks': int(sum(ranking_df['rank_range'] <= 5)),
            'sequences_with_high_variance': int(sum(ranking_df['rank_range'] > 20)),
            'max_rank_range': int(ranking_df['rank_range'].max()),
            'mean_rank_range': float(ranking_df['rank_range'].mean())
        }
    }
    
    # Save results
    output_dir = f"/mnt/results/{job_name}"
    ranking_df.to_csv(f"{output_dir}/ranking_analysis.csv", index=False)
    training_stats.to_csv(f"{output_dir}/training_metrics.csv", index=False)
    
    with open(f"{output_dir}/summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Analysis complete!")
    print(f"Mean final R²: {summary_stats['mean_final_r2']:.4f} ± {summary_stats['std_final_r2']:.4f}")
    print(f"Mean ranking std: {summary_stats['ranking_stability']['mean_rank_std']:.2f}")
    print(f"Sequences with stable ranks (≤5 positions): {summary_stats['ranking_stability']['sequences_with_stable_ranks']}")
    print(f"Max rank range: {summary_stats['ranking_stability']['max_rank_range']}")
    
    return summary_stats

@app.function(
    image=image,
    volumes={"/mnt/results": results_volume},
    timeout=14400,
)
def run_multi_training(job_name: str, num_runs: int = 10):
    """Run multiple training runs sequentially with different seeds."""
    # Generate different seeds for each run
    base_seed = 42
    seeds = [base_seed + i * 100 for i in range(num_runs)]
    
    print(f"Starting {num_runs} training runs for job '{job_name}' (sequential)")
    print(f"Seeds: {seeds}")
    
    # Run training sequentially
    results = []
    for run_id in range(1, num_runs + 1):
        seed = seeds[run_id - 1]
        print(f"\nStarting run {run_id}/{num_runs} (seed {seed})")
        
        result = train_single_run.remote(run_id, seed, job_name)
        results.append(result)
        print(f"✓ Run {run_id} completed successfully")
    
    print(f"\nTraining complete! All {num_runs} runs succeeded")
    return results

@app.function(
    image=image,
    volumes={"/mnt/results": results_volume},
    timeout=14400,
)
def generate_all_predictions(job_name: str, num_runs: int = 10):
    """Generate predictions for all trained models sequentially."""
    print(f"Generating predictions for all {num_runs} models...")
    
    # Run predictions sequentially
    all_results = []
    for run_id in range(1, num_runs + 1):
        print(f"Generating predictions for run {run_id}/{num_runs}")
        
        result = generate_predictions_for_run.remote(run_id, job_name)
        all_results.extend(result)
        print(f"✓ Predictions for run {run_id} completed")
    
    print(f"All predictions generated!")
    return all_results

@app.function(
    image=image,
    volumes={"/mnt/results": results_volume},
    timeout=36000,  # 10 hours total
)
def run_full_pipeline_remote(job_name: str, num_runs: int = 10):
    """Run the complete pipeline remotely in detached mode."""
    print(f"Starting full pipeline for job '{job_name}' with {num_runs} runs")
    
    # Phase 1: Training
    print("Phase 1: Training models...")
    run_multi_training.remote(job_name, num_runs)
    print("Training phase completed!")
    
    # Phase 2: Generate predictions
    print("Phase 2: Generating predictions...")
    generate_all_predictions.remote(job_name, num_runs)
    print("Predictions phase completed!")
    
    # Phase 3: Analysis
    print("Phase 3: Analyzing ranking stability...")
    results = analyze_ranking_stability.remote(job_name, num_runs)
    print("Analysis phase completed!")
    
    print(f"Full pipeline completed successfully for job '{job_name}'!")
    print("Results saved to Modal volume 'cnn-training-results'")
    
    return results

@app.local_entrypoint()
def train(job_name: str, num_runs: int = 10):
    """Run training only."""
    print(f"Starting multi-training for job: {job_name}")
    results = run_multi_training.remote(job_name, num_runs)
    print("Training complete!")
    return results

@app.local_entrypoint()
def predict(job_name: str, num_runs: int = 10):
    """Generate predictions only."""
    print(f"Generating predictions for job: {job_name}")
    results = generate_all_predictions.remote(job_name, num_runs)
    print("Predictions complete!")
    return results

@app.local_entrypoint()
def analyze(job_name: str, num_runs: int = 10):
    """Analyze ranking stability only."""
    print(f"Analyzing ranking stability for job: {job_name}")
    results = analyze_ranking_stability.remote(job_name, num_runs)
    print("Analysis complete!")
    return results

@app.local_entrypoint()
def full_pipeline(job_name: str, num_runs: int = 10):
    """Run full pipeline in detached mode - returns immediately."""
    print(f"Launching full pipeline for job: {job_name} (detached mode)")
    
    # This runs everything remotely and returns immediately
    # The pipeline will continue running even if you disconnect
    job = run_full_pipeline_remote.spawn(job_name, num_runs)
    
    print(f"Pipeline launched! Job ID: {job.object_id}")
    print("The pipeline will run completely in the cloud.")
    print("Check Modal dashboard for progress, or run:")
    print(f"modal logs {job.object_id}")
    
    return job