from collections import defaultdict
from tqdm import tqdm, trange

import numpy as np
from sklearn.metrics import r2_score
import torch
from torch import nn
import torch.optim as optim


def train_variant_cnn(model, train_loader, val_loader, config, device, updates=True):
    """
    Main training function with all the bells and whistles
    """
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min'
    )
    
    criterion = nn.MSELoss()
    
    if updates:
        pbar = trange(config.epochs)
    else:
        pbar = range(config.epochs)

    for epoch in pbar:
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(embeddings).flatten()
            loss = criterion(outputs, labels)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            # Store predictions and losses
            train_losses.append(loss.item())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = model(embeddings).flatten()
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        if updates:
            pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_loss:.4f}"
            })

    return model


def train_up_to_dist(model, data_holder, dist):
    dh_train = data_holder.for_cut_offs(max_distance=dist)
    train_loader, val_loader = dh_train.train_val_split()
    return train_variant_cnn(model, train_loader, val_loader, config, device, updates=False)


def train_on_dist(model, data_holder, dist):
    dh_train = data_holder.for_cut_offs(min_distance=dist, max_distance=dist)
    train_loader, val_loader = dh_train.train_val_split()
    return train_variant_cnn(model, train_loader, val_loader, config, device, updates=False)


def train_on_per(model, data_holder, per):
    dh_train = data_holder.for_per_of_data(per=per)
    train_loader, val_loader = dh_train.train_val_split()
    return train_variant_cnn(model, train_loader, val_loader, config, device, updates=False)


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