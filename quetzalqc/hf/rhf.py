from dataclasses import dataclass

import torch

from quetzalqc.core.diis import DIIS
from quetzalqc.core.density import rhf_density
from quetzalqc.core.energy import rhf_total_energy
from quetzalqc.core.jk import JKBuilder
from quetzalqc.core.linalg import solve_gen_eig


@dataclass
class RHFConfig:
    max_iter: int = 50
    e_conv: float = 1e-10
    d_conv: float = 1e-8
    use_diis: bool = True
    diis_start: int = 2
    diis_size: int = 6
    damping: float = 0.0


class RHF:
    def __init__(
        self,
        S: torch.Tensor,
        H: torch.Tensor,
        eri: torch.Tensor,
        enuc: float,
        nelec: int,
        cfg: RHFConfig,
    ):
        self.S = S
        self.H = H
        self.enuc = enuc
        self.nelec = nelec
        self.nocc = nelec // 2
        self.cfg = cfg

        self.jk = JKBuilder(eri)
        self.diis = DIIS(cfg.diis_size) if cfg.use_diis else None

    def initial_guess(self) -> torch.Tensor:
        _, C = solve_gen_eig(self.H, self.S)
        return rhf_density(C, self.nocc)

    def build_fock(self, P: torch.Tensor) -> torch.Tensor:
        J, K = self.jk.build(P)
        return self.H + J - 0.5 * K

    def run(self, initial_density: torch.Tensor | None = None) -> dict:
        cfg = self.cfg
        if self.diis is not None:
            self.diis.focks.clear()
            self.diis.errs.clear()

        if initial_density is None:
            P = self.initial_guess()
        else:
            P = initial_density.detach().clone()
        E_old = None
        E = None

        for it in range(1, cfg.max_iter + 1):
            F = self.build_fock(P)

            if self.diis is not None and it >= cfg.diis_start:
                self.diis.push(F, P, self.S)
                F_diis = self.diis.extrapolate()
                if F_diis is not None:
                    F = F_diis

            eps, C = solve_gen_eig(F, self.S)
            P_new = rhf_density(C, self.nocc)

            if cfg.damping > 0:
                P_new = (1.0 - cfg.damping) * P_new + cfg.damping * P

            dP = torch.linalg.norm(P_new - P).item()
            E = rhf_total_energy(P_new, self.H, F, self.enuc)
            dE = abs(E - E_old) if E_old is not None else float("inf")

            print(f"iter {it:02d}  E = {E:+.12f}  dE = {dE:.3e}  dP = {dP:.3e}")

            if dE < cfg.e_conv and dP < cfg.d_conv:
                return {
                    "converged": True,
                    "iterations": it,
                    "E": E,
                    "eps": eps,
                    "C": C,
                    "P": P_new,
                }

            P = P_new
            E_old = E

        return {"converged": False, "iterations": cfg.max_iter, "E": E, "P": P}
