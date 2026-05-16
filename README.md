# Neuro-Vault – Deep Generative Brain Health Modeling

**NeuroVault** is a research‑grade demonstration of the [`neurodatasets`](https://pypi.org/project/neurodatasets/) package.  
It goes **far beyond simple classification** by training a **Variational Autoencoder (VAE)** on the `sleep_study_college` dataset to:

- Learn a compressed **latent representation** of 18 brain‑health‑related variables.
- Reconstruct original features with **high fidelity** (average **R² = 0.79**).
- Detect **anomalous subjects** via reconstruction probability – useful for spotting atypical brain‑behaviour profiles.
- Generate **synthetic brain profiles** that respect multivariate correlations.
- Visualise the latent space with **t‑SNE** and anomaly flags.

All data loading and curation is handled by a single call to `neurodatasets`.

---

## Why This Is a Strong Demonstration

| Traditional Classification | NeuroVault (this project) |
|----------------------------|----------------------------|
| Supervised, needs labels | Fully unsupervised |
| Low accuracy on small datasets | High reconstruction fidelity (R² up to **0.93** per feature) |
| Outputs a single class | Outputs anomaly scores, synthetic samples, and latent visualisations |
| Ignores data structure | Captures the full multivariate distribution |
| Uses one dataset | Can be extended to multiple `neurodatasets` modalities |

---

## Results (from 3000‑epoch training)

**Average reconstruction R²:** 0.788

| Feature             | R²    |
|---------------------|-------|
| GPA                 | 0.674 |
| ClassesMissed       | 0.763 |
| CognitionZscore     | 0.716 |
| PoorSleepQuality    | 0.719 |
| DepressionScore     | 0.831 |
| AnxietyScore        | 0.815 |
| StressScore         | 0.850 |
| DASScore            | 0.903 |
| Happiness           | 0.753 |
| Drinks              | 0.695 |
| WeekdayBed          | 0.797 |
| WeekdayRise         | 0.799 |
| WeekdaySleep        | 0.841 |
| WeekendBed          | 0.723 |
| WeekendRise         | 0.755 |
| WeekendSleep        | 0.746 |
| AverageSleep        | 0.874 |
| AllNighter          | 0.925 |

The model excels on core mental‑health and sleep variables (DASScore, StressScore, AverageSleep, AllNighter) and still performs well on noisier features like GPA.

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python -m src.main
```

The script will:
- Load the data from `neurodatasets`.
- Train a VAE for 3000 epochs.
- Print reconstruction R² per feature.
- Flag the top‑10% most anomalous subjects.
- Save `reconstruction.png` (original vs. reconstructed profiles).
- Save `latent_space.png` (t‑SNE with anomaly flags).
- Print three synthetic brain profiles.

---

## How It Works

1. **Data loading & preprocessing**  
   `nd.load_dataset('sleep_study_college')` loads 253 student records with 27 columns. Categorical/ID columns are dropped, leaving 18 numeric features. Missing values are imputed, and features are standardised.

2. **VAE architecture**  
   - Encoder: 18 → 64 → 32 → 8 (latent dim)  
   - Decoder: 8 → 32 → 64 → 18  
   - Loss: MSE + KL divergence (β=1)

3. **Training**  
   The model is trained for 3000 epochs with the Adam optimiser (learning rate=0.001).

4. **Evaluation**  
   - Reconstruction R² per feature and overall mean.  
   - Anomaly scores (reconstruction error per subject).  
   - Latent space projected with t‑SNE, coloured by anomaly status.

5. **Generation**  
   Random samples from the standard Gaussian prior are decoded into realistic brain‑health profiles.

---

## Dependencies

See [`requirements.txt`](requirements.txt) for exact versions. Core libraries:

- `neurodatasets` – for effortless data access.
- `torch` – deep learning framework.
- `scikit-learn` – preprocessing and t‑SNE.
- `numpy`, `pandas`, `matplotlib`, `seaborn` – data handling and plotting.

---

## Disclaimer

This project is for **research and educational purposes only**. The generated profiles and anomaly flags are not clinical diagnoses. Always consult qualified professionals for medical decisions.

