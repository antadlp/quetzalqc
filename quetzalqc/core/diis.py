import torch


class DIIS:
    """
    DIIS extrapolation on Fock with commutator error:
      e = F P S - S P F
    """

    def __init__(self, max_vec: int = 6):
        self.max_vec = max_vec
        self.focks: list[torch.Tensor] = []
        self.errs: list[torch.Tensor] = []

    def push(self, F: torch.Tensor, P: torch.Tensor, S: torch.Tensor) -> None:
        e = F @ P @ S - S @ P @ F
        self.focks.append(F.detach().clone())
        self.errs.append(e.detach().clone())
        if len(self.focks) > self.max_vec:
            self.focks.pop(0)
            self.errs.pop(0)

    def extrapolate(self) -> torch.Tensor | None:
        n = len(self.focks)
        if n < 2:
            return None

        B = torch.empty((n + 1, n + 1), dtype=self.focks[0].dtype)
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0

        for i in range(n):
            for j in range(n):
                B[i, j] = torch.sum(self.errs[i] * self.errs[j])

        rhs = torch.zeros(n + 1, dtype=self.focks[0].dtype)
        rhs[-1] = -1

        try:
            coeff = torch.linalg.solve(B, rhs)[:-1]
        except RuntimeError:
            return None

        Fnew = torch.zeros_like(self.focks[0])
        for c, F in zip(coeff, self.focks):
            Fnew += c * F
        return Fnew
