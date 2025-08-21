from tqdm import trange

import numpy as np
from sklearn.metrics import r2_score
import torch
from torch import nn
import torch.optim as optim


def train_variant_cnn(model, train_loader, val_loader, config, device, updates=True):
    """
    Main training function with comprehensive metrics tracking
    """
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min'
    )
    
    criterion = nn.MSELoss()
    
    # Training history tracking
    train_losses = []
    val_losses = []
    val_r2_history = []
    
    if updates:
        pbar = trange(config.epochs)
    else:
        pbar = range(config.epochs)

    for epoch in pbar:
        # Training phase
        model.train()
        epoch_train_losses = []
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings).flatten()
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_losses = []
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = model(embeddings).flatten()
                loss = criterion(outputs, labels)
                
                epoch_val_losses.append(loss.item())
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = np.mean(epoch_val_losses)
        val_losses.append(val_loss)
        
        # Calculate R²
        val_r2 = r2_score(val_labels, val_preds)
        val_r2_history.append(val_r2)
        
        scheduler.step(val_loss)
        
        if updates:
            pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_loss:.4f}",
                'val_r2': f"{val_r2:.4f}"
            })

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_r2_history': val_r2_history,
        'final_r2': val_r2_history[-1]
    }



def r2_score_for_model_and_loader(model, loader):
    model.eval()

    predictions = []
    actuals = []
    with torch.no_grad():
        for xbatch, ybatch in loader:
            predictions.append(
                model(xbatch.to("cuda")).flatten()
            )
            actuals.append(ybatch)

    predictions = torch.concat(predictions).cpu().detach().numpy()
    actuals = torch.concat(actuals).cpu().detach().numpy()

    return r2_score(actuals, predictions)