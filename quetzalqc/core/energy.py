import torch


def rhf_total_energy(
    P: torch.Tensor, H: torch.Tensor, F: torch.Tensor, enuc: float
) -> float:
    """E = 0.5 * Tr[P(H + F)] + E_nuc."""
    E_elec = 0.5 * torch.sum(P * (H + F))
    return float(E_elec + enuc)
