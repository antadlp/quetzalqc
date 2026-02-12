# quetzalqc

Modular quantum chemistry prototype with a PySCF integral backend and Torch SCF core.

## Layout

- `quetzalqc/core`: linear algebra, density, J/K, energy, DIIS
- `quetzalqc/hf`: RHF driver and HF backends
- `quetzalqc/dft`: DFT placeholders
- `quetzalqc/md`: MD placeholders
- `quetzalqc/cuda`: CUDA placeholders
- `tests`: pytest validation
- `examples`: runnable scripts
- `docs`: documentation notes

## Quick start

```bash
pip install -r requirements.txt -r requirements-dev.txt
python examples/run_h2.py
python -m pytest -q
python examples/scan_h2.py
```
