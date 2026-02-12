import torch


class JKBuilder:
    """
    ERI in chemist notation (mu nu|lambda sigma):
      J_{mu nu} = sum_{lambda sigma} P_{lambda sigma} (mu nu|lambda sigma)
      K_{mu nu} = sum_{lambda sigma} P_{lambda sigma} (mu lambda|nu sigma)
    """

    def __init__(self, eri: torch.Tensor):
        self.eri = eri

    def build(self, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eri = self.eri
        J = torch.einsum("mnls,ls->mn", eri, P)
        K = torch.einsum("mlns,ls->mn", eri, P)
        return J, K
