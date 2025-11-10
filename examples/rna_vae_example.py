"""
Example script showing how to use RNAVAE with the integrated embkit BaseVAE.

This replaces your standalone TensorFlow RNA VAE script with the PyTorch
version integrated into embkit's infrastructure.
"""
import os
import pandas as pd
import torch
from embkit.models.vae import RNAVAE
from embkit import get_device


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Load your cancer data (example structure)
    # Replace this with your actual data loading
    data_path = "path/to/your/cancer_data.csv"  # or parquet, feather, etc.
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable with your actual data file.")
        return
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Assuming:
    # - First column is cancer type
    # - Remaining columns are gene features
    cancer_col = df.columns[0]
    feature_cols = df.columns[1:].tolist()
    
    print(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    print(f"Cancer types: {df[cancer_col].nunique()}")
    
    # Initialize RNAVAE with your feature list
    print("\nInitializing RNAVAE...")
    model = RNAVAE(
        features=feature_cols,
        latent_dim=768,
        lr=0.0005
    )
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Train the model
    print("\nTraining RNAVAE...")
    X = df[feature_cols]
    
    history = model.fit(
        X=X,
        epochs=100,
        batch_size=512,
        kappa=1.0,  # Beta warmup rate (0 -> 1)
        early_stopping_patience=3,
        device=device,
        progress=True
    )
    
    # Save model
    model_path = "rna_vae_model.pt"
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    
    # Generate embeddings for each cancer type
    print("\nGenerating embeddings per cancer type...")
    model.eval()
    
    cancer_types = df[cancer_col].unique()
    
    for cancer in cancer_types:
        # Filter data for this cancer
        cancer_data = df[df[cancer_col] == cancer][feature_cols]
        
        # Get embeddings
        with torch.no_grad():
            X_tensor = torch.FloatTensor(cancer_data.values).to(device)
            mu, logvar, z = model.encoder(X_tensor)
            embeddings = mu.cpu().numpy()
        
        # Save to CSV
        output_file = f"embeddings_{cancer}.csv"
        embeddings_df = pd.DataFrame(
            embeddings,
            columns=[f"dim_{i}" for i in range(embeddings.shape[1])]
        )
        embeddings_df.to_csv(output_file, index=False)
        print(f"Saved {cancer}: {embeddings_df.shape[0]} samples -> {output_file}")
    
    print("\n✓ Complete!")
    print(f"Training history keys: {list(history.keys())}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final beta: {history['beta'][-1]:.4f}")


if __name__ == "__main__":
    main()
