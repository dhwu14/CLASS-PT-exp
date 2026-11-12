# CLASS-PT-exp: Exact-time fading dark sector extension of CLASS-PT

**CLASS-PT-exp** is a modified version of [CLASS-PT](https://github.com/Michalychforever/CLASS-PT), built on the [CLASS](https://github.com/lesgourg/class_public) Boltzmann solver, for computing large-scale-structure observables in [the fading dark sector model (FDS)](https://arxiv.org/abs/1906.08261) which has been recently also analyzed in [arXiv:2507.03090](https://arxiv.org/abs/2507.03090).

The extension evolves baryon and coupled-dark-matter perturbations separately, using time-dependent perturbation-theory kernels instead of the usual Einstein–de Sitter (EdS) time-factorization approximation. It computes one-loop matter and biased-tracer power spectra, including redshift-space multipoles, and supports two-fluid bias operators, EFT counterterms and stochastic contributions. The implementation also includes redshift-space distortions (RSD), Alcock–Paczynski (AP) mapping and infrared (IR) resummation. Optional primordial non-Gaussianity templates are available, subject to the model and IR-resummation settings.

## Getting started

### Requirements

- C/C++ compiler with OpenMP support (GCC recommended)
- [OpenBLAS](https://www.openblas.net/) (configure its location in `Makefile`)
- Python 3, NumPy and Cython for the optional `classy` interface

### Build

Configure the `OPENBLAS` path in `Makefile`, then run from the repository root:

```bash
make class
# Optional Python interface:
make classy
```

For additional build details, see the [upstream CLASS-PT instructions](https://github.com/Michalychforever/CLASS-PT/blob/master/instructions.pdf) and the [CLASS repository](https://github.com/lesgourg/class_public).

### Example FDS run

The file [`qcdm_eft_fullshape.ini`](qcdm_eft_fullshape.ini) provides an example of the coupled FDS calculation with the exact-time kernel branch and high-resolution numerical settings used in an earlier LCDM-limit comparison.

```bash
./class qcdm_eft_fullshape.ini
```

The supplied example has nonzero FDS couplings and illustrative tracer-bias parameters; it is not a fitted cosmology. For a decoupled LCDM-limit check within the FDS branch, set:

```ini
qscf_parameters_2 = 0.0
qscf_parameters_5 = 0.0
```

Keep the remaining background and numerical settings matched to the reference calculation. The example's resolution settings are not a universal convergence guarantee.

## Acknowledgments

This project is derived from CLASS-PT; it is not an independent implementation of the underlying CLASS/CLASS-PT solver. Please acknowledge the original CLASS and CLASS-PT authors and cite the original CLASS-PT code paper, [arXiv:2004.10607](https://arxiv.org/abs/2004.10607), when using this software. The upstream CLASS-PT authors include Mikhail Ivanov, Anton Chudaykin, Marko Simonović, Oliver Philcox and Giovanni Cabass.

Please also consult the upstream repositories for applicable licensing and attribution requirements.

### Authors

- [David Wu](mailto:davidhzwu@gmail.com)
- Georges Obied
- Cumrun Vafa
