import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data import load_and_preprocess
from src.model import VAE
from src.train import train
from src.evaluate import reconstruction_score, anomaly_scores
from src.visualize import plot_reconstruction, plot_latent_space, generate_synthetic
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('NeuroVault')

def main():
    # 1. Load data
    X_scaled, scaler, feature_names = load_and_preprocess()
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. Build model
    vae = VAE(input_dim=len(feature_names), latent_dim=8)

    # 3. Train
    logger.info("Training VAE...")
    vae = train(vae, loader, epochs=3000, lr=1e-3, log_interval=30, logger=logger)

    # 4. Evaluate
    r2_mean, r2_per_feat = reconstruction_score(vae, loader)
    logger.info(f"Average reconstruction R²: {r2_mean:.4f}")
    for feat, r2 in zip(feature_names, r2_per_feat):
        logger.info(f"  {feat:20s}: {r2:.4f}")

    anom = anomaly_scores(vae, loader)
    top5 = np.argsort(anom)[-5:]
    logger.info(f"Top‑5 anomalous subjects: {top5.tolist()}")
    logger.info(f"Their scores: {anom[top5].round(4)}")

    # 5. Visualise
    plot_reconstruction(vae, X_tensor, scaler, feature_names)
    plot_latent_space(vae, loader, anom)
    profiles = generate_synthetic(vae, scaler, feature_names)
    logger.info("Generated synthetic profiles:")
    for i, prof in enumerate(profiles):
        logger.info(f"  Profile {i+1}:")
        for k, v in prof.items():
            logger.info(f"    {k:20s}: {v:.2f}")

if __name__ == '__main__':
    main()
