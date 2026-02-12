import torch


def symm_orthogonalizer(S: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    """Compute X = S^(-1/2) with eigen-decomposition."""
    w, U = torch.linalg.eigh(S)
    w = w.clamp_min(eps)
    return U @ torch.diag(w.rsqrt()) @ U.T


def solve_gen_eig(F: torch.Tensor, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve FC = S C eps using symmetric orthogonalization."""
    X = symm_orthogonalizer(S)
    Fp = X.T @ F @ X
    eps, Cp = torch.linalg.eigh(Fp)
    C = X @ Cp
    return eps, C
