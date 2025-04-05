import librosa
import numpy as np
import soundfile as sf
from sklearn.decomposition import NMF

def compute_beta_divergence(V, W, H, beta):
    """Обчислення β-дивергенції між V та W@H."""
    WH = np.dot(W, H)
    if beta == 2:
        return np.sum((V - WH) ** 2)
    elif beta == 1:
        return np.sum(V * np.log(V / WH) - V + WH)
    elif beta == 0:
        return np.sum(V / WH - np.log(V / WH) - 1)
    else:
        return np.sum((V ** beta + (beta - 1) * WH ** beta - beta * V * WH ** (beta - 1)) / (beta * (beta - 1)))

def nmf_with_beta_divergence(V, n_components, beta, max_iter=500):
    """NMF з використанням β-дивергенції."""
    model = NMF(n_components=n_components, beta_loss=beta, solver='mu', max_iter=max_iter, random_state=42)
    W = model.fit_transform(V)
    H = model.components_
    return W, H

def separate_sources(y, sr, n_components=2, beta=1, max_iter=500):
    """Розділення джерел з використанням NMF та маскування."""
    # Обчислення STFT
    S = np.abs(librosa.stft(y))
    phase = np.angle(librosa.stft(y))

    # Застосування NMF з β-дивергенцією
    W, H = nmf_with_beta_divergence(S, n_components, beta, max_iter)

    # Реконструкція джерел з використанням маскування
    sources = []
    for i in range(n_components):
        Vi = np.outer(W[:, i], H[i])
        mask = Vi / (np.dot(W, H) + 1e-8)
        Si = mask * S
        Si_complex = Si * np.exp(1j * phase)
        y_i = librosa.istft(Si_complex)
        sources.append(y_i)
        sf.write(f'source_{i+1}.wav', y_i, sr)
        print(f"Збережено: source_{i+1}.wav")

    return sources

# Приклад використання
y, sr = librosa.load('final_combined_pipe_and_voice.wav', sr=None, mono=True)
separate_sources(y, sr, n_components=2, beta=1, max_iter=1000)
