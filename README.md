# QuetzalQC

**QuetzalQC** is a modular, research-oriented quantum chemistry engine built in Python using PyTorch for linear algebra and designed for extensibility toward DFT, post-HF methods, and molecular dynamics.

> **Note:** GitHub's Markdown renderer does not natively display LaTeX equations. The math below uses `$$...$$` syntax which is supported by GitHub via KaTeX, but you may need to view this file in a renderer that understands MathJax/KaTeX (e.g. a local Markdown viewer or GitHub Pages with math enabled) to see properly formatted equations.

The current implementation provides a fully modular **Restricted Hartree–Fock (RHF)** solver validated against reference implementations.

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

Within the Born–Oppenheimer approximation, the electronic Hamiltonian is:

$$
\hat{H}_{elec} =
\sum_i \left(
-\frac{1}{2} \nabla_i^2
- \sum_A \frac{Z_A}{|\mathbf{r}_i - \mathbf{R}_A|}
\right)
+
\sum_{i<j} \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}
$$

Exact solutions are intractable for multi-electron systems.

---

## Restricted Hartree–Fock Approximation

The wavefunction is approximated as a single Slater determinant.

Molecular orbitals are expanded in a finite basis:

$$
\phi_i(\mathbf{r}) =
\sum_\mu C_{\mu i} \chi_\mu(\mathbf{r})
$$

The Roothaan equations are:

$$
\mathbf{F} \mathbf{C} =
\mathbf{S} \mathbf{C} \bm{\varepsilon}
$$

where:

- **S** = overlap matrix
- **F** = Fock matrix
- **C** = MO coefficients

---

## Fock Matrix

$$
F_{\mu\nu} =
H_{\mu\nu}
+
J_{\mu\nu}
-
\frac{1}{2} K_{\mu\nu}
$$

Coulomb matrix:

$$
J_{\mu\nu} =
\sum_{\lambda\sigma}
P_{\lambda\sigma}
(\mu\nu|\lambda\sigma)
$$

Exchange matrix:

$$
K_{\mu\nu} =
\sum_{\lambda\sigma}
P_{\lambda\sigma}
(\mu\lambda|\nu\sigma)
$$

Density matrix:

$$
P_{\mu\nu} =
2 \sum_{i}^{occ}
C_{\mu i} C_{\nu i}
$$

---

## Self-Consistent Field (SCF)

The RHF procedure iteratively solves:

$$
\mathbf{P}^{(n)}
\rightarrow
\mathbf{F}^{(n)}
\rightarrow
\mathbf{C}^{(n)}
\rightarrow
\mathbf{P}^{(n+1)}
$$

until:

$$
\|\mathbf{P}^{(n+1)} - \mathbf{P}^{(n)}\| < \delta
$$

---

## Total Energy

$$
E =
\frac{1}{2}
\mathrm{Tr}
\left[
\mathbf{P}
(\mathbf{H} + \mathbf{F})
\right]
+ E_{nuc}
$$

---

# Software Architecture

```

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

````

---

## Module Responsibilities

### Backend Layer
Provides:

- Overlap matrix S
- One-electron Hamiltonian H
- Two-electron integrals (μν|λσ)
- Nuclear repulsion energy
- Electron count

The SCF driver is backend-agnostic.

---

### Linear Algebra Layer
Solves the generalized eigenvalue problem:

$$
\mathbf{F} \mathbf{C} =
\mathbf{S} \mathbf{C} \bm{\varepsilon}
$$

using symmetric orthogonalization.

---

### Density Module
Builds:

$$
\mathbf{P} =
2 \mathbf{C}_{occ} \mathbf{C}_{occ}^T
$$

---

### JK Module
Handles the computational bottleneck:

$$
O(N^4)
$$

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
````

Clone the repository:

```bash
git clone https://github.com/yourusername/quetzalqc.git
cd quetzalqc
```

---

# Example: H₂ / STO-3G

```bash
python run_h2.py
```

Expected output:

* Convergence in a few iterations
* Energy consistent with PySCF reference

---

# Validation

Unit tests compare energies against PySCF:

```bash
python -m pytest -q
```

---

# Computational Scaling

The dominant cost is:

$$
O(N^4)
$$

from Coulomb and exchange contractions.

Future acceleration strategies:

* Density fitting (RI)
* GPU tensor kernels
* CUTLASS integration
* Screening techniques

---

# Roadmap

Planned extensions:

* LDA / GGA DFT
* Density fitting
* Analytical gradients
* CUDA acceleration
* Born–Oppenheimer molecular dynamics
* Differentiable electronic structure

---

# Philosophy

QuetzalQC is designed as:

* A transparent implementation of quantum chemistry algorithms
* A research platform for experimentation
* A bridge between physics, mathematics, and high-performance computing

---

# License

(Choose your license here, e.g., MIT or Apache 2.0)

---

# Author

Antonio Alvarez
