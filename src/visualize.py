import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_reconstruction(model, X_tensor, scaler, feature_names, n_samples=3, save_path='reconstruction.png'):
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(X_tensor[:n_samples])
    orig = scaler.inverse_transform(X_tensor[:n_samples].numpy())
    recon = scaler.inverse_transform(recon.numpy())
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, n_samples * 2))
    for i in range(n_samples):
        axes[i, 0].bar(feature_names, orig[i], color='steelblue')
        axes[i, 0].set_title(f'Original subject {i+1}')
        axes[i, 0].tick_params(axis='x', rotation=45)
        axes[i, 1].bar(feature_names, recon[i], color='darkorange')
        axes[i, 1].set_title(f'Reconstructed subject {i+1}')
        axes[i, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_latent_space(model, data_loader, anomaly_scores, save_path='latent_space.png'):
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0]
            mu, _ = model.encode(x)
            latents.append(mu.numpy())
    latents = np.vstack(latents)
    lat_2d = TSNE(n_components=2, random_state=42).fit_transform(latents)
    plt.figure(figsize=(8, 6))
    threshold = np.percentile(anomaly_scores, 90)
    is_anom = anomaly_scores > threshold
    plt.scatter(lat_2d[~is_anom, 0], lat_2d[~is_anom, 1], c='steelblue', alpha=0.6, label='Normal')
    plt.scatter(lat_2d[is_anom, 0], lat_2d[is_anom, 1], c='crimson', alpha=0.8, label='Anomalous (top 10%)')
    plt.legend()
    plt.title('Latent space (t‑SNE) with anomaly flags')
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_synthetic(model, scaler, feature_names, n_samples=3):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim)
        synthetic = model.decode(z).numpy()
    synthetic = scaler.inverse_transform(synthetic)
    profiles = []
    for i, row in enumerate(synthetic):
        profile = dict(zip(feature_names, row))
        profiles.append(profile)
    return profiles
