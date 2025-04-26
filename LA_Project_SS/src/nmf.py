import numpy as np
import matplotlib.pyplot as plt
import librosa

def enhanced_initialization(data_matrix, rank):
    '''
    Enhanced initialization strategy based on SVD properties
    Optimized for audio spectrograms
    '''
    u, s, v = np.linalg.svd(data_matrix, full_matrices=False)
    v = v.T
    
    w = np.zeros((data_matrix.shape[0], rank))
    h = np.zeros((rank, data_matrix.shape[1]))
    
    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(v[:, 0])

    for i in range(1, rank):
        u_i = u[:, i]
        v_i = v[:, i]
        
        # Decompose into positive and negative parts
        u_pos = np.maximum(u_i, 0)
        u_neg = np.maximum(-u_i, 0)
        v_pos = np.maximum(v_i, 0)
        v_neg = np.maximum(-v_i, 0)
        
        # Calculate norms
        u_pos_norm = np.linalg.norm(u_pos, 2)
        u_neg_norm = np.linalg.norm(u_neg, 2)
        v_pos_norm = np.linalg.norm(v_pos, 2)
        v_neg_norm = np.linalg.norm(v_neg, 2)
        
        # Calculate projection magnitudes
        pos_magnitude = u_pos_norm * v_pos_norm
        neg_magnitude = u_neg_norm * v_neg_norm
        
        epsilon = 1e-12
        if pos_magnitude >= neg_magnitude:
            w[:, i] = np.sqrt(s[i] * pos_magnitude) / (u_pos_norm + epsilon) * u_pos
            h[i, :] = np.sqrt(s[i] * pos_magnitude) / (v_pos_norm + epsilon) * v_pos
        else:
            w[:, i] = np.sqrt(s[i] * neg_magnitude) / (u_neg_norm + epsilon) * u_neg
            h[i, :] = np.sqrt(s[i] * neg_magnitude) / (v_neg_norm + epsilon) * v_neg
    
    return w, h


def calculate_cost(V, W, H, cost_type='kld'):
    '''
    Calculate the cost function between original and reconstructed matrices
    KL divergence is particularly effective for audio spectra
    '''
    epsilon = 1e-12
    reconstruction = W @ H
    
    if cost_type == 'frobenius':
        return 0.5 * np.sum((V - reconstruction)**2)
    elif cost_type == 'kld':
        return np.sum(V * np.log((V + epsilon) / (reconstruction + epsilon)) - V + reconstruction)
    elif cost_type == 'is':
        return np.sum(V / (reconstruction + epsilon) - np.log(V / (reconstruction + epsilon)) - 1)
    else:
        raise ValueError(f"Unsupported cost_type: {cost_type}")


def speech_nmf(V, rank, max_iterations=2000, threshold=1e-8, cost_type='kld', sparsity=0.1):
    '''
    Enhanced NMF optimized for speech separation with multiple cost functions
    and sparsity constraints
    '''
    V = np.maximum(V, 0)
    W, H = enhanced_initialization(V, rank)
    
    cost_history = []
    epsilon = 1e-12
    
    for iteration in range(max_iterations):
        if cost_type == 'frobenius':
            numerator = W.T @ V
            denominator = W.T @ (W @ H) + sparsity + epsilon
            H *= numerator / denominator
            
            numerator = V @ H.T
            denominator = (W @ H) @ H.T + epsilon
            W *= numerator / denominator
            
        elif cost_type == 'kld':
            WH = W @ H + epsilon
            H *= (W.T @ (V / WH)) / (W.T @ np.ones_like(V) + sparsity + epsilon)
            W *= ((V / WH) @ H.T) / (np.ones_like(V) @ H.T + epsilon)
            
        elif cost_type == 'is':
            WH = W @ H + epsilon
            H *= (W.T @ (V / (WH**2))) / (W.T @ (1.0 / WH) + sparsity + epsilon)
            W *= ((V / (WH**2)) @ H.T) / ((1.0 / WH) @ H.T + epsilon)
        else:
            raise ValueError(f"Unsupported cost_type: {cost_type}")
        
        # Normalize W columns and scale H accordingly
        column_norms = np.linalg.norm(W, axis=0)
        column_norms[column_norms == 0] = 1.0
        W /= column_norms
        H = np.diag(column_norms) @ H
        
        current_cost = calculate_cost(V, W, H, cost_type)
        cost_history.append(current_cost)
        
        if iteration > 0 and abs(cost_history[-2] - current_cost) < threshold * cost_history[-2]:
            print(f"Converged at iteration {iteration}")
            break
    
    return W, H, cost_history


def NMF(V, rank, num_sources, clustering_method='frequency', **nmf_params):
    '''
    Separate audio sources by clustering NMF components
    
    Parameters:
    -----------
    V : np.ndarray
        Input spectrogram (magnitude)
    rank : int
        Number of NMF components
    num_sources : int
        Number of sources to separate
    clustering_method : str
        Method to assign components to sources ('frequency' supported)
    nmf_params : dict
        Additional parameters for speech_nmf
    
    Returns:
    --------
    source_spectrograms : list of np.ndarray
        List of separated source spectrograms
    W : np.ndarray
        Basis matrix
    H : np.ndarray
        Activation matrix
    component_assignments : np.ndarray
        Array assigning each component to a source index
    '''
    W, H, _ = speech_nmf(V, rank, **nmf_params)
    
    component_assignments = np.zeros(rank, dtype=int)
    
    if clustering_method == 'frequency':
        frequencies = np.arange(W.shape[0])[:, np.newaxis]
        centroids = np.sum(frequencies * W, axis=0) / (np.sum(W, axis=0) + 1e-12)
        
        sorted_indices = np.argsort(centroids)
        components_per_source = rank // num_sources
        
        for i in range(num_sources):
            start_idx = i * components_per_source
            end_idx = (i + 1) * components_per_source if i < num_sources - 1 else rank
            component_assignments[sorted_indices[start_idx:end_idx]] = i
    
    source_spectrograms = []
    for i in range(num_sources):
        source_indices = np.where(component_assignments == i)[0]
        W_source = W[:, source_indices]
        H_source = H[source_indices, :]
        source_spec = W_source @ H_source
        source_spectrograms.append(source_spec)
    
    return source_spectrograms, W, H, component_assignments