import pytest
import torch
from pyscf import gto, scf

from quetzalqc.hf.pyscf_backend import MoleculeSpec, PySCFBackend
from quetzalqc.hf.rhf import RHF, RHFConfig


def pyscf_rhf_energy(atom: str, basis: str = "sto-3g") -> float:
    mol = gto.M(atom=atom, basis=basis, spin=0, charge=0, unit="Angstrom")
    mf = scf.RHF(mol)
    return float(mf.kernel())


@pytest.mark.parametrize("R", [0.5, 0.74, 1.5])
def test_h2_sto3g_rhf_matches_pyscf(R: float) -> None:
    torch.set_default_dtype(torch.float64)

    atom = f"H 0 0 0; H 0 0 {R}"
    spec = MoleculeSpec(atom=atom, basis="sto-3g", charge=0, spin=0, unit="Angstrom")
    be = PySCFBackend(spec)

    cfg = RHFConfig(
        max_iter=80,
        e_conv=1e-12,
        d_conv=1e-10,
        use_diis=True,
        diis_start=2,
        diis_size=8,
        damping=0.0,
    )

    rhf = RHF(S=be.S, H=be.H, eri=be.eri, enuc=be.enuc, nelec=be.nelec, cfg=cfg)
    out = rhf.run()

    assert out["converged"], "RHF did not converge."

    e_ref = pyscf_rhf_energy(atom, "sto-3g")
    e_ours = out["E"]
    assert abs(e_ours - e_ref) < 1e-8, f"Mismatch: ours={e_ours}, pyscf={e_ref}"
