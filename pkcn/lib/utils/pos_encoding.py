import torch
import itertools
import numpy as np
from scipy.linalg import orthogonal_procrustes

def compute_procrustes_distance(x, y):
    """
    Compute Procrustes distance between two tensors of shape [B, V, J, 3]
    using only PyTorch (no numpy)
    """
    B, V, J, D = x.shape
    dist = torch.zeros(B, device=x.device)

    for b in range(B):
        X = x[b].reshape(-1, D)  # [V*J, 3]
        Y = y[b].reshape(-1, D)

        # Center
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # Normalize
        norm_X = torch.norm(X)
        norm_Y = torch.norm(Y)
        if norm_X > 0: X = X / norm_X
        if norm_Y > 0: Y = Y / norm_Y

        # SVD for optimal rotation R
        M = X.T @ Y
        U, _, Vh = torch.linalg.svd(M)
        R = U @ Vh

        aligned_X = X @ R
        frob_dist = torch.norm(aligned_X - Y, p='fro')
        dist[b] = frob_dist

    return dist

def piecewise_index(x, alpha=1, beta=9, gamma=2000, dtype=torch.long):
    """
    Quantize float distances into [-beta, ..., 0, ..., +beta] integer buckets.
    """
    sign = torch.sign(x)
    abs_x = torch.abs(x)
    index = torch.zeros_like(x)

    mask_close = abs_x <= alpha
    index[mask_close] = abs_x[mask_close].round()

    mask_far = ~mask_close
    log_val = (torch.log(abs_x[mask_far] / alpha + 1e-6) / np.log(gamma)) + alpha
    index[mask_far] = log_val.round().clamp(max=beta)

    return (sign * index).to(dtype)

def build_srpe_matrix(multiview_kp3d, alpha=1, beta=9, gamma=2000):
    """
    Strucutre-aware relative positional encoding
    """
    B, N, V, J, D = multiview_kp3d.shape
    dist_matrix = torch.zeros(B, N, N)
    
    for i, j in itertools.combinations(range(N), 2) :
        d = compute_procrustes_distance(multiview_kp3d[:, i], multiview_kp3d[:, j])
        dist_matrix[:, i, j] = d
        dist_matrix[:, j, i] = d
        
    srpe_index = piecewise_index(dist_matrix, alpha=alpha, beta=beta, gamma=gamma)
    return srpe_index

if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, V, J, D = 2, 4, 5, 17, 3  # small test case
    data = torch.rand(B, N, V, J, D)

    srpe_index = build_srpe_matrix(data)

    print("SRPE index shape:", srpe_index.shape)
    print("SRPE index:\n", srpe_index)
    print("dtype:", srpe_index.dtype)
    print("min/max:", srpe_index.min().item(), srpe_index.max().item())

    # Optional assertions
    assert srpe_index.shape == (B, N, N)
    assert srpe_index.dtype == torch.long
    assert torch.all(srpe_index == srpe_index.transpose(1, 2))  # symmetry
    for b in range(B):
        assert torch.all(srpe_index[b].diagonal() == 0)  # zero diagonal