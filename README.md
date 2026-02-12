# QuetzalQC

**QuetzalQC** is a modular, research-oriented quantum chemistry engine built in Python using PyTorch for linear algebra and designed for extensibility toward DFT, post-HF methods, and molecular dynamics.

The current implementation provides a fully modular **Restricted Hartree-Fock (RHF)** solver validated against reference implementations.

---

## Vision

QuetzalQC is designed as:

- A transparent educational quantum chemistry engine
- A research platform for method development
- A differentiable-ready electronic structure framework
- A foundation for future DFT and ab initio molecular dynamics

The guiding principles are:

- Mathematical clarity
- Modular architecture
- Backend abstraction
- Scalability toward GPU acceleration

---

# Mathematical Overview

## Electronic Structure Problem

Within the Born-Oppenheimer approximation, the electronic Hamiltonian is:

```text
H_elec =
sum_i [
  -(1/2) nabla_i^2
  - sum_A Z_A / |r_i - R_A|
]
+ sum_{i<j} 1 / |r_i - r_j|
```

Exact solutions are intractable for multi-electron systems.

---

## Restricted Hartree-Fock Approximation

The wavefunction is approximated as a single Slater determinant.

Molecular orbitals are expanded in a finite basis:

```text
phi_i(r) = sum_mu C_{mu i} chi_mu(r)
```

The Roothaan equations are:

```text
F C = S C epsilon
```

where:

- **S** = overlap matrix
- **F** = Fock matrix
- **C** = MO coefficients

---

## Fock Matrix

```text
F_{mu nu} = H_{mu nu} + J_{mu nu} - (1/2) K_{mu nu}
```

Coulomb matrix:

```text
J_{mu nu} = sum_{lambda sigma} P_{lambda sigma} (mu nu | lambda sigma)
```

Exchange matrix:

```text
K_{mu nu} = sum_{lambda sigma} P_{lambda sigma} (mu lambda | nu sigma)
```

Density matrix:

```text
P_{mu nu} = 2 * sum_{i in occ} C_{mu i} C_{nu i}
```

---

## Self-Consistent Field (SCF)

The RHF procedure iteratively solves:

```text
P^(n) -> F^(n) -> C^(n) -> P^(n+1)
```

until:

```text
||P^(n+1) - P^(n)|| < delta
```

---

## Total Energy

```text
E = (1/2) * Tr[ P (H + F) ] + E_nuc
```

---

# Software Architecture

```text
quetzalqc/
  backends/
    pyscf_backend.py
  core/
    linalg.py
    density.py
    jk.py
    energy.py
    diis.py
    rhf.py
  tests/
  examples/
```

---

## Module Responsibilities

### Backend Layer

Provides:

- Overlap matrix S
- One-electron Hamiltonian H
- Two-electron integrals (mu nu | lambda sigma)
- Nuclear repulsion energy
- Electron count

The SCF driver is backend-agnostic.

---

### Linear Algebra Layer

Solves the generalized eigenvalue problem:

```text
F C = S C epsilon
```

using symmetric orthogonalization.

---

### Density Module

Builds:

```text
P = 2 C_occ C_occ^T
```

---

### JK Module

Handles the computational bottleneck:

```text
O(N^4)
```

via explicit tensor contractions.

---

### SCF Driver

Orchestrates:

1. Initial guess
2. Fock construction
3. Diagonalization
4. Density update
5. Convergence check

---

# Installation

```bash
pip install torch pyscf
```

Clone the repository:

```bash
git clone https://github.com/yourusername/quetzalqc.git
cd quetzalqc
```

---

# Example: H2 / STO-3G

```bash
python run_h2.py
```

Expected output:

- Convergence in a few iterations
- Energy consistent with PySCF reference

---

# Validation

Unit tests compare energies against PySCF:

```bash
python -m pytest -q
```

---

# Computational Scaling

The dominant cost is:

```text
O(N^4)
```

from Coulomb and exchange contractions.

Future acceleration strategies:

- Density fitting (RI)
- GPU tensor kernels
- CUTLASS integration
- Screening techniques

---

# Roadmap

Planned extensions:

- LDA / GGA DFT
- Density fitting
- Analytical gradients
- CUDA acceleration
- Born-Oppenheimer molecular dynamics
- Differentiable electronic structure

---

# Philosophy

QuetzalQC is designed as:

- A transparent implementation of quantum chemistry algorithms
- A research platform for experimentation
- A bridge between physics, mathematics, and high-performance computing

---

# License

(Choose your license here, e.g., MIT or Apache 2.0)

---

# Author

Antonio Alvarez
