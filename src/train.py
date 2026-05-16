import torch
import torch.optim as optim
import torch.nn as nn

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss

def train(model, data_loader, epochs=150, lr=1e-3, log_interval=30, logger=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            x = batch[0]
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if logger and (epoch + 1) % log_interval == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} – Loss: {total_loss/len(data_loader):.4f}")
    return model
