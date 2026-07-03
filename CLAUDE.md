# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable, with optional dependencies):**
```bash
pip install -e .
pip install -e .[class,camb,jax,extras]   # with Boltzmann codes and JAX
```

**Run all tests:**
```bash
pytest cosmoprimo/tests/
pytest cosmoprimo/emulators/tests/
pytest cosmoprimo/emulators/tools/tests/
```

**Run a single test file or function:**
```bash
pytest cosmoprimo/tests/test_cosmology.py
pytest cosmoprimo/tests/test_cosmology.py::test_params
pytest cosmoprimo/emulators/tests/test_base.py::test_jax
```

**Build docs:**
```bash
cd doc && make html
```

## Architecture

### Cosmology + Engine + Section pattern

`Cosmology` (`cosmology.py`) is the top-level parameter container. It does not compute anything by itself — it delegates all calculations to an **engine** and **sections**.

- `Cosmology(engine='class', h=0.67, omega_cdm=0.12, ...)` creates a cosmology and binds an engine.
- An engine (`BaseEngine` subclass, registered via `RegisteredEngine` metaclass) wraps a Boltzmann solver or analytic approximation.
- Each engine exposes **sections** — named groups of outputs: `Background`, `Thermodynamics`, `Primordial`, `Perturbations`, `Transfer`, `Harmonic`, `Fourier`. The list is defined in `_Sections` in `cosmology.py`.
- Sections are lazy: `cosmo.get_background()` instantiates the `Background` object for the current engine on demand. Repeated calls return the cached instance.
- `Cosmology.__getattr__` auto-routes attribute access to the right section: `cosmo.comoving_radial_distance` resolves to `cosmo.get_background().comoving_radial_distance`. This only works when the attribute belongs to exactly one section.

**To add a new engine:** create a module (e.g. `myengine.py`) with a class that inherits `BaseEngine` and sets `name = 'myengine'`. Import it lazily in `get_engine()` in `cosmology.py`. Implement the relevant section classes (`Background`, `Fourier`, etc.) inside the same module.

### Available engines

| Engine string | Module | Notes |
|---|---|---|
| `'class'` / `'classy'` | `classy.py` | Requires `pyclass` |
| `'camb'` | `camb.py` | Requires `camb` |
| `'eisenstein_hu'` | `eisenstein_hu.py` | Analytic, JAX-differentiable |
| `'eisenstein_hu_nowiggle'` | `eisenstein_hu_nowiggle.py` | Analytic, JAX-differentiable |
| `'bbks'` | `bbks.py` | Analytic, JAX-differentiable |
| `'astropy'` | `astropy.py` | Requires `astropy` |
| `'tabulated'` | `tabulated.py` | Interpolates pre-saved tables |
| `'axiclass'`, `'mochiclass'`, `'negnuclass'` | respective `.py` | CLASS variants |
| `'isitgr'`, `'mgcamb'` | respective `.py` | Modified gravity |
| `'capse'`, `'cosmopower_bolliet2023'` | `emulators/` | JAX neural-network emulators |

### Parameter system

- **Conflict groups** (`_conflict_parameters_no_alias`): only one parameter per group may be provided (e.g. cannot pass both `sigma8` and `A_s`). `merge_params` / `check_params` enforce this.
- **Aliases** (`_alias_parameters`): `H0`→`h`, `ombh2`→`omega_b`, `omch2`→`omega_cdm`, `logA`→`A_s`, `w`→`w0_fld`, etc. Normalisation to the internal `(h, Omega_*, A_s/sigma8)` basis happens in `Cosmology._compile_params`.
- `cosmo['omega_cdm']` (i.e. `__getitem__`) retrieves both base parameters and derived quantities (Omega fractions, `N_eff`, `theta_cosmomc`, etc.).
- `cosmo.clone(base='input', h=0.7)` returns a new `Cosmology` updating from the original input parameter basis. `base='internal'` updates in the `(h, Omega)` basis instead.
- `cosmo.solve(param, func, target)` bisects to find the value of `param` such that `func(cosmo) == target`.

### JAX support

`jax.py` provides the JAX/NumPy dispatch layer:
- `numpy_jax(*args)` returns `jax.numpy` if any arg is a JAX array/tracer, else `numpy`.
- `use_jax(*args)` / `use_jax(*args, tracer_only=True)` detect JAX context.
- All core classes (`Cosmology`, `BaseEngine`, BAO filters, interpolators) register themselves as JAX pytrees via `register_pytree_node_class`, implementing `tree_flatten` / `tree_unflatten`. Numerical parameters become leaves; non-numerical (strings, lists of modes, z grids) go into aux_data.
- Boltzmann solvers (`class`, `camb`) are **not** JAX-differentiable. Analytic engines (`bbks`, `eisenstein_hu*`) and emulators are.

### Key utilities

- **`interpolator.py`**: `PowerSpectrumInterpolator2D(k, z, pk)` / `CorrelationFunctionInterpolator2D` — log-log cubic spline interpolation with power-law extrapolation in k. Also compute `sigma(r,z)` via `TophatVariance`.
- **`fftlog.py`**: `PowerToCorrelation`, `CorrelationToPower`, `TophatVariance` — FFTlog via `numpy.fft` or `pyfftw`.
- **`bao_filter.py`**: `PowerSpectrumBAOFilter(method)` / `CorrelationFunctionBAOFilter(method)` — multiple methods (Wallish2018, Brieden2022, EHNoWiggle, Kirkby2013, etc.). JAX-differentiable methods noted in module docstring.
- **`fiducial.py`**: Pre-built cosmologies — `DESI` (= `AbacusSummitBase`), `AbacusSummit(name)`, `Planck2018FullFlatLCDM`, `BOSS`, `Uchuu`, `TabulatedDESI`, `DESIDR2Flatw0waCDM`.
- **`emulators/`**: See dedicated section below.

### Persistence

`cosmo.save(fn)` / `Cosmology.load(fn)` serialize via `numpy.save` / `numpy.load` on a plain Python dict (`__getstate__` / `__setstate__`). Only the parameter dict and engine name (not the engine's computed cache) are saved; the engine is re-initialized on load.

## Emulators (`cosmoprimo/emulators/`)

The emulator subsystem has two layers:

- **`tools/`** — a generic, calculator-agnostic emulation toolkit.
- **`emulators/`** (top level) — cosmoprimo-specific wiring that turns a `Cosmology` into a calculator and exposes trained models as `BaseEngine` subclasses.

### tools/ layer

**`Samples`** (`tools/samples.py`): dictionary of arrays (`X.param` for inputs, `Y.section.quantity` for outputs). Supports slicing, concatenation, MPI scatter/gather, save/load as `.npy` or `.npz`.

**Samplers** (`tools/samples.py`): `InputSampler`, `GridSampler`, `DiffSampler`, `QMCSampler`. Each calls a `calculator(**params) → dict` and accumulates a `Samples` object. `QMCSampler` supports quasi-random engines: `'sobol'`, `'halton'`, `'lhs'`, `'rqrs'`. Use `sampler.run(niterations=N, save_fn=fn)` to run and persist incrementally.

**`Emulator`** (`tools/base.py`): orchestrates sampling, fitting, and prediction.
- `set_engine(engine)` — maps output name wildcards (e.g. `'fourier.*'`) to a `BaseEmulatorEngine`.
- `set_samples(samples, engine, ...)` — applies `xoperation` / `yoperation` transforms to samples, then initializes per-output engines.
- `fit()` — fits each engine to its samples.
- `predict(params)` — runs all engines, reverses `yoperation`, returns dict.
- `save(fn)` / `Emulator.load(fn)` — serialize to `.npy` via `__getstate__` / `__setstate__`.

**`BaseEmulatorEngine` subclasses** (registered via `RegisteredEmulatorEngine` metaclass):
- `TaylorEmulatorEngine` (`tools/taylor.py`) — finite-difference Taylor expansion up to configurable order/accuracy. Generates samples on a sparse grid automatically via `get_default_samples`.
- `MLPEmulatorEngine` (`tools/mlp.py`) — MLP trained with Flax + optax (cosine LR schedule with 10% linear warmup). Key params: `nhidden=(32,32,32)`, `activation='silu'`, `loss='mse'`. Auto-appends `ScaleOperation` if no normalisation operation present.
- `PointEmulatorEngine` — returns a fixed point (useful for testing).

**`Operation`** (`tools/base.py`): pre/post-processing transform with `initialize(v)`, `__call__(v, X=)`, `inverse(v, X=)`. Built-in operations: `ScaleOperation`, `NormOperation`, `Log10Operation`, `ArcsinhOperation`, `PCAOperation`, `ChebyshevOperation`. Custom operations can be created with `Operation(expr_str, inverse=expr_str)` where `expr_str` is an expression string evaluated with `v` (output array) and `X` (input param dict) in scope. Operations are serializable.

### emulators/ layer (cosmoprimo-specific)

**`get_calculator(cosmo, section=None)`** (`emulators/__init__.py`): wraps a `Cosmology` into a plain `calculator(**params) → dict` function. It calls `cosmo.clone(**params)`, iterates over requested sections, and calls each section's `__getstate__` (as defined in `emulated.py`) to extract a flat dict of named arrays. This is the bridge between `Cosmology` and the generic `Emulator` / samplers.

**`EmulatedEngine`** (`emulators/emulated.py`): a `BaseEngine` subclass that loads a trained `Emulator` (one `.npy` file, or a `path` dict of multiple files merged via `emulator.update()`). At `__init__`, it calls the emulator's internal engines for each section and stores a `_predict(section)` callable. Sigma8/A_s mismatch between cosmology and emulator is handled by `_needs_rescale` and `_rescale_sigma8()`.

Section classes in `emulated.py` (`Background`, `Thermodynamics`, `Harmonic`, `Fourier`) define the I/O contract:
- `__getstate__` specifies which quantities are exported during training (e.g. `Background.__getstate__` saves `z` + `rho_ncdm`, `p_ncdm`, `rho_fld`, `time`, `comoving_radial_distance` at a fixed z-grid).
- `__setstate__` rebuilds 1D interpolators from those arrays for inference.

**Concrete emulator engines** (`emulators/hybrid.py`):
- `CAPSEEngine` — loads thermodynamics (camb) + harmonic/Fourier (jaxcapse) emulators; auto-downloads from GitHub if `.npy` files missing.
- `CosmopowerBolliet2023Engine` — loads cosmopower-based thermodynamics + harmonic + Fourier emulators.
- `camb_mnu_w_wa_cmb_engine` — thermodynamics + harmonic only.
- Emulator files are stored under `emulators/train/` (gitignored except pre-converted ones); `COSMOPRIMO_EMULATOR_DIR` env var overrides the search path.

**Cosmoprimo-specific operations** (`emulators/__init__.py`):
- `FourierNormOperation` — normalises all power spectra by the `delta_cb × delta_cb` spectrum, separating k-shape (z=0) from z-evolution; reduces dynamic range for fitting.
- `HarmonicNormOperation` — divides Cl by A_s and shifts the ell axis by the ratio of `theta_cosmomc` to a fiducial value; improves emulator accuracy near acoustic peaks.

**Conversion scripts** (`emulators/conversion.py`): import external emulator weights (jaxcapse `.eqx` files, cosmopower TensorFlow models) and write them as cosmoprimo `Emulator` `.npy` files. Run as `__main__` script; not part of the importable API.

### Training workflow

Reference scripts are in `emulators/train/` (e.g. `train_classy.py`):

```python
# 1. Sample
cosmo = DESI(engine='class', neutrino_hierarchy='degenerate')
calculator = get_calculator(cosmo, section=['background', 'thermodynamics', 'fourier'])
sampler = QMCSampler(calculator, params={'h': (0.5, 0.9), 'omega_cdm': (0.03, 0.3), ...},
                     engine='lhs', seed=42, save_fn='samples_0_100000.h5')
sampler.run(save_every=100, niterations=100000)

# 2. Fit
samples = Samples.concatenate([Samples.read(fn) for fn in glob.glob('samples_*.h5')])
emulator = Emulator()
emulator.set_samples(samples=samples,
                     engine={'thermodynamics.*': MLPEmulatorEngine(nhidden=(10,)*5, activation='tanh'),
                             'fourier.*': MLPEmulatorEngine(nhidden=(512,)*3)},
                     yoperation=FourierNormOperation())
emulator.fit()
emulator.write('emulator.h5')

# 3. Use
cosmo = DESI(engine=EmulatedEngine.read('emulator.h5'))
cosmo.comoving_radial_distance(z=1.)
jax.jacfwd(lambda p: DESI(**p, engine=EmulatedEngine.read('emulator.h5')).comoving_radial_distance(1.))(params)
```

### Emulator tests

```bash
pytest cosmoprimo/emulators/tests/test_base.py    # end-to-end: sample → fit → predict → JAX grad
pytest cosmoprimo/emulators/tests/test_mlp.py
pytest cosmoprimo/emulators/tests/test_taylor.py
pytest cosmoprimo/emulators/tools/tests/          # generic toolkit unit tests
```
