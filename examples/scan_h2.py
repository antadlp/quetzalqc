import csv
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running this script directly from the examples/ folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quetzalqc.hf.pyscf_backend import MoleculeSpec, PySCFBackend
from quetzalqc.hf.rhf import RHF, RHFConfig

try:
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


def run_single_point(
    R: float, cfg: RHFConfig, initial_density: torch.Tensor | None = None
) -> dict:
    atom = f"H 0 0 0; H 0 0 {R}"
    spec = MoleculeSpec(atom=atom, basis="sto-3g", charge=0, spin=0, unit="Angstrom")
    be = PySCFBackend(spec)
    rhf = RHF(S=be.S, H=be.H, eri=be.eri, enuc=be.enuc, nelec=be.nelec, cfg=cfg)
    out = rhf.run(initial_density=initial_density)
    out["R"] = R
    return out


def main() -> None:
    torch.set_default_dtype(torch.float64)

    Rs = np.linspace(0.4, 3.0, 27)
    cfg = RHFConfig(
        max_iter=80,
        e_conv=1e-10,
        d_conv=1e-8,
        use_diis=True,
        diis_start=2,
        diis_size=8,
        damping=0.0,
    )

    out_path = Path("h2_sto3g_rhf_scan.csv")
    rows = []

    prev_density = None
    for R in Rs:
        print("\n" + "=" * 60)
        print(f"R = {R:.3f} Angstrom")
        out = run_single_point(float(R), cfg, initial_density=prev_density)
        prev_density = out["P"].detach().clone() if out["converged"] else None
        rows.append(
            {
                "R_angstrom": out["R"],
                "E_hartree": out["E"],
                "converged": out["converged"],
                "iterations": out["iterations"],
            }
        )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["R_angstrom", "E_hartree", "converged", "iterations"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path.resolve()}")

    if HAS_PLOT:
        rvals = [r["R_angstrom"] for r in rows if r["converged"]]
        evals = [r["E_hartree"] for r in rows if r["converged"]]
        plt.figure(figsize=(6, 4))
        plt.plot(rvals, evals, marker="o")
        plt.xlabel("R (Angstrom)")
        plt.ylabel("E (Hartree)")
        plt.title("H2 RHF / STO-3G bond scan")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
