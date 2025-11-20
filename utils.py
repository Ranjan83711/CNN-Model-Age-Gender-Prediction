# src/utils.py
import os
import torch

def save_checkpoint(state, path):
    """
    Saves a checkpoint dictionary to a file.
    Creates the directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, device='cpu'):
    """
    Loads a checkpoint returning a dictionary with keys like:
    - model_state
    - optimizer_state
    - epoch
    - val_loss
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)
