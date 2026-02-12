from dataclasses import dataclass

import torch
from pyscf import gto

torch.set_default_dtype(torch.float64)


@dataclass
class MoleculeSpec:
    atom: str
    basis: str
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"


class PySCFBackend:
    """
    Provides AO-basis integrals for RHF:
      S, H = T+V, ERI (mu nu|lambda sigma), nuclear repulsion, nelec.
    """

    def __init__(self, spec: MoleculeSpec):
        self.mol = gto.M(
            atom=spec.atom,
            basis=spec.basis,
            charge=spec.charge,
            spin=spec.spin,
            unit=spec.unit,
        )
        self.nao = self.mol.nao_nr()
        self.nelec = int(sum(self.mol.nelec))
        self.enuc = float(self.mol.energy_nuc())

        s = self.mol.intor("int1e_ovlp")
        t = self.mol.intor("int1e_kin")
        v = self.mol.intor("int1e_nuc")
        eri = self.mol.intor("int2e")

        self.S = torch.from_numpy(s)
        self.H = torch.from_numpy(t + v)
        self.eri = torch.from_numpy(eri)
