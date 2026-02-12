import torch

from quetzalqc.hf.pyscf_backend import MoleculeSpec, PySCFBackend
from quetzalqc.hf.rhf import RHF, RHFConfig

torch.set_default_dtype(torch.float64)


def main() -> None:
    spec = MoleculeSpec(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )
    be = PySCFBackend(spec)

    cfg = RHFConfig(
        max_iter=50,
        e_conv=1e-10,
        d_conv=1e-8,
        use_diis=True,
        diis_start=2,
        diis_size=6,
        damping=0.0,
    )

    rhf = RHF(S=be.S, H=be.H, eri=be.eri, enuc=be.enuc, nelec=be.nelec, cfg=cfg)
    out = rhf.run()

    print("\nConverged:", out["converged"])
    print("Iterations:", out["iterations"])
    print("Total E  :", out["E"])


if __name__ == "__main__":
    main()
