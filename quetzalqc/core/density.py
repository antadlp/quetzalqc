import torch


def rhf_density(C: torch.Tensor, nocc: int) -> torch.Tensor:
    """P = 2 C_occ C_occ^T for closed-shell RHF."""
    Cocc = C[:, :nocc]
    return 2.0 * (Cocc @ Cocc.T)
