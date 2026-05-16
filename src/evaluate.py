import torch
import numpy as np

def reconstruction_score(model, data_loader):
    model.eval()
    all_x, all_recon = [], []
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0]
            recon, _, _ = model(x)
            all_x.append(x.numpy())
            all_recon.append(recon.numpy())
    X_true = np.vstack(all_x)
    X_recon = np.vstack(all_recon)
    r2s = []
    for i in range(X_true.shape[1]):
        ss_res = np.sum((X_true[:, i] - X_recon[:, i]) ** 2)
        ss_tot = np.sum((X_true[:, i] - X_true[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        r2s.append(r2)
    return np.mean(r2s), r2s

def anomaly_scores(model, data_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0]
            recon, _, _ = model(x)
            mse = torch.mean((x - recon) ** 2, dim=1).numpy()
            scores.extend(mse)
    return np.array(scores)
