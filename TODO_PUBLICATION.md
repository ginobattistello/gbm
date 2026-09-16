# Pre-publication to-do — GBM Toolbox

Working document for the cleanup pass before first public release.
Audited 2026-09-16 against branch `jax` at `5fc3dad`.

Decisions taken: import package renamed to `gbmtoolbox`; failing tests and
scripts ported to JAX (not reverted to NumPy).

Documentation target (revised 2026-09-16): the final documentation set is two
files, written after the code is green so they describe a working API.

1. **`docs/VALIDATION.md`** — scientific validation of the toolbox, driven by
   the reference scripts in `dev/`: what each check asks, the reference
   calculation, and the measured result.
2. **`docs/MANUAL.md`** — what the toolbox is for: parameter estimation in a
   generative modelling framework, static and dynamic models, via MAP
   estimation and filtering.

Both carry the mathematics with references, kept simple and readable for
newcomers rather than written for specialists. `DEV.md` and `CONTRIBUTING.md`
fold into this set.

---

## What the audit found

The `jax` commit (`4bff5da`) rewrote the core to a JAX backend and **added** a
new generation of tests and validation scripts. It did not modify a single
pre-existing test file. The old NumPy-era tests and scripts were left in place,
untouched, beside the new ones.

The split is exact, with no exceptions:

| Generation | Files | Result |
|---|---|---|
| JAX-era (added by `4bff5da`) | 6 test files, 3 `validate_*.py` | **9/9 tests pass, 3/3 scripts pass** |
| NumPy-era (untouched) | 8 test files, 7 numbered `dev/0*.py` | **18 tests fail, 5 scripts fail** |

Every failure is the same root cause. The backend now requires traceable model
functions; the old files still use `x.copy()`, `x[c] += …`, `int(y_t)`,
`np.array([…])` inside model functions. `bayesgbm/examples/tutorial_models.py`
was converted and all 8 tutorials pass — **it is the reference idiom for every
port below**.

Three findings are more serious than the idiom mismatch and drive the ordering.

### Finding 1 — thirteen validation checks were dropped

`validation.py` lost 13 error paths in the rewrite. Diff of pre/post error
strings. These are **not** all obsolete; each needs a deliberate decision:

| Dropped check | Assessment |
|---|---|
| `latent_uncertainty='filtered' requires initial-state uncertainty or process_covariance` | **Silent regression — restore.** See Finding 2. |
| `evolution must be deterministic for fixed inputs` | Likely still valid; confirm against JAX tracing. |
| `categorical observation must return a probability vector summing to one` | Probably obsolete — `families.py` now normalizes internally. Confirm. |
| `categorical y contains a category outside the returned probability vector` | Still a real user error. Assess for restore. |
| `Bernoulli observation must return scalar P(y=1) in [0, 1]` | **Correctly obsolete** — contract changed to logit. See Finding 3. |
| `hard_bounds must contain one (low, high) pair per static parameter` | Moved, not lost → `optimization.py:173`. No action. |
| `subject {n}: Gaussian y must be numeric` | Assess. |
| `subject {n}: observation must return numeric output` | Partly covered by finiteness guards at `validation.py:116`. Assess. |
| `subject {n}: observation returned NaN/Inf at prior means` | Covered by `validation.py:116`. Likely no action. |
| `subject {n}: prior-mean model evaluation returned invalid log likelihood` | Covered by `validation.py:168`. Likely no action. |
| `parameter names are inconsistent with prior dimension` | Assess — may have moved to `priors.py`. |
| 2 × module docstrings | Not checks. No action. |

### Finding 2 — `filtered` on a deterministic model is no longer rejected

The string `requires initial-state uncertainty` existed at
`validation.py:145` before the rewrite and **exists nowhere in the package
now**. `Config` still accepts `latent_uncertainty="filtered"` and
`individual_fit.py:227` still routes to `_latent_filtered`, so a deterministic
model no longer raises — it silently produces whatever the filter returns with
zero process noise.

This is a scientific regression, not a test problem. Fixing the failing test by
relaxing its regex would conceal it.

### Finding 3 — the README documents the wrong Bernoulli contract

`families.py:23` applies `jax.nn.sigmoid(eta)` internally, so the observation
function must now return a **logit**. `tutorial_models.py` is correct
(`return beta * (x[1] - x[0])  # Bernoulli logit`).

`README.md:45` still documents `Bernoulli: P(y=1)`, and the minimal example at
`README.md:72` returns `sigmoid(beta * (x[1] - x[0]))`. A user following the
README **double-applies the sigmoid**: the model still runs and still fits, but
is silently wrong. This is the highest-severity documentation defect in the
repo.

### Other blocking facts

- **`jax` is not a declared dependency.** Imported by 9 modules including
  `__init__.py`; absent from `pyproject.toml`. A clean install cannot import
  the package, which means the CI matrix (3.10–3.13) has never actually
  exercised the JAX core.
- **`docs/VALIDATION_RESULTS.md` claims "38 passed" and "all seven scripts
  completed"** against a NumPy/SciPy environment with no JAX. Unearned
  validation claim; must be regenerated from a real run.
- **No document mentions JAX anywhere.**
- `RELEASE_CHECKLIST.md` is staged for deletion. Its content is still useful —
  fold the relevant items into Phase 7 rather than losing them.

---

## Phase 0 — Safety net

- [ ] `git stash` or commit the staged `RELEASE_CHECKLIST.md` deletion so the
      tree is clean before work starts
- [ ] Branch from current state: `git checkout -b cleanup/publication`
- [ ] Record the baseline in the branch description: 29 passed / 18 failed,
      `dev/` 5 of 10 failing

## Phase 1 — Make the dependency real

Must come first: until this is done, CI results are meaningless.

- [ ] Add `jax>=0.4` and `jaxlib>=0.4` to `[project.dependencies]` in
      `pyproject.toml` (installed locally: 0.11.0 — set the floor deliberately,
      not to match the dev box)
- [ ] Verify in a clean venv that `pip install -e .` then `import bayesgbm`
      succeeds
- [ ] Re-run the suite; confirm the 29/18 split is unchanged (this phase must
      not alter behaviour)
- [ ] Commit

## Phase 2 — Restore the dropped validation checks — **DONE** (`ebfb5ac`)

Reading the full pre/post modules corrected Finding 1: the string diff
overstated the loss. Nine of the thirteen were **rewritten for the logit
contract, not dropped** — Bernoulli/categorical logit shape (`:120`, `:123`),
categorical range (`:124`), numeric output (`:113`), NaN/Inf (`:116`),
prior-mean likelihood (`:168`, now covering value *and* gradient), hard_bounds
(`:66`), parameter names (`:56`). Non-numeric Gaussian `y` is caught earlier
and more strictly by `_check_numeric_finite` at `:88`.

Two were genuine losses, both restored:

- [x] **`filtered` on a deterministic model.** Not a crash — the path returned
      an all-zero state covariance as `sd=0.0` at every trial, labelled
      `uncertainty_type="filtered"`, `method="deterministic"`. Plausible-looking
      output a user would plot as real credible bands. Restored before the JAX
      smoke test so the scientific error is not masked by a tracer error.
- [x] **Evolution determinism.** Verified by experiment that JAX does *not*
      catch a host-side RNG in `evolution`: it is sampled once and frozen into
      the compiled graph. Restored as a double call and comparison.
- [x] Both covered by new tests using JAX-native models, independent of the
      `conftest.py` fixtures Phase 3 will port
- [x] Suite 31 passed / 18 failed; failing IDs unchanged; 8/8 tutorials pass

`test_validation.py::test_filtered_uncertainty_rejected_for_deterministic_model`
still fails, correctly: its NumPy fixture fails the per-subject JAX check first.
It will pass once Phase 3 ports `conftest.py`.

## Phase 3 — Port the tests to JAX — **DONE** (`5acb83f`)

Porting `conftest.py` alone cleared 10 of 18. Final: **49 passed, 0 failed.**
Three items were contract changes rather than idiom: the logit contract in
fixtures and reference cases; a string label in `u` (now numeric/bool only);
and `filtering_method` renamed to `generalized_gaussian_fisher`.

## Phase 4 — Port the `dev/` scripts — **DONE** (`7993320`)

All 10 scripts pass. Measured results match the pre-JAX record: Bernoulli
quadrature 1.231e-03 (~1.2e-3), categorical 4.136e-03 (~4.1e-3), propagated
0.552% (~0.55%), identifiability rank 1/2.

The Kalman check reads ~2.3e-10 against ~1e-13 recorded. **Not a regression** —
it is `Config.filter_jitter` (default 1e-9), scaling linearly: 1e-12 jitter
gives 2.3e-13, jitter off gives 1.1e-16 (machine precision), matching the
pre-JAX closed-form EKF. Iteration budget ruled out first; error is flat from 4
to 50 iterations.

**Open for your decision:** `03_linear_gaussian_filter.py` and
`validate_linear_gaussian_filter.py` genuinely overlap — both compare the
filter to a hand-written scalar Kalman recursion at `atol=1e-7`. `03` uses 4
fixed trials with an observation offset and prints errors; `validate_` uses 25
random trials without an offset and only asserts. Recommend keeping `03` (it
reports numbers, which VALIDATION.md needs) and dropping `validate_`, but not
acting without your say.

## ~~Phase 3 — Port the tests to JAX~~ (original plan below)

Convert `tests/conftest.py` first; several failures resolve through the shared
fixtures. Use `examples/tutorial_models.py` as the reference.

```python
# tests/conftest.py — before
x = x.copy()
x[c] += alpha * (float(u_t["reward"]) - x[c])

# after
return x.at[c].add(alpha * (u_t["reward"] - x[c]))
```

- [ ] `tests/conftest.py` — `binary_model`, `gaussian_filter_model` fixtures
- [ ] `tests/test_state_model_likelihoods.py` — 4 failures
- [ ] `tests/test_individual_fit.py` — 4 failures
- [ ] `tests/test_filtering.py` — 3 failures
- [ ] `tests/test_diagnostics.py` — 2 failures
- [ ] `tests/test_predictive_bms_hbi.py` — 2 failures
- [ ] `tests/test_additional_contracts.py` — 2 failures
- [ ] `tests/test_validation.py` — 1 failure (should now pass via Phase 2)
- [ ] Full suite green: 47 passed

**Do not delete the old test files as redundant.** Coverage overlaps only
partially. `test_filtered_discrete.py` (new) tests *AD-Hessian support*;
`test_filtering.py` (old) tests *Bernoulli-vs-quadrature and categorical PSD*
reference cases. Different questions — deleting the old files would silently
drop the scientific reference checks.

One genuine overlap: `test_linear_gaussian_filter.py::test_unified_filter_matches_scalar_kalman`
vs `test_filtering.py::test_linear_gaussian_filter_matches_kalman`. Resolve
after both pass.

## Phase 4 — Port the `dev/` scripts

- [ ] `01_likelihood_reference_cases.py` — family likelihoods vs formulas
- [ ] `03_linear_gaussian_filter.py` — filter vs exact Kalman
- [ ] `04_bernoulli_filter_quadrature.py` — Bernoulli vs quadrature
- [ ] `05_categorical_filter_quadrature.py` — categorical vs quadrature
- [ ] `07_information_identifiability.py` — local identifiability on a ridge
- [ ] Report overlap between numbered `0*.py` and newer `validate_*.py`
      (notably `03_` vs `validate_linear_gaussian_filter.py`) and recommend a
      resolution — **do not delete anything without asking**
- [ ] All 10 scripts pass

**If a reference case's numerical error has shifted under JAX float64, that is a
scientific finding — report it, do not adjust the tolerance to make it pass.**

## Phase 5 — Write the documentation set — **DONE** (`c149ff0`)

`docs/MANUAL.md` and `docs/VALIDATION.md` written; `METHODS.md`,
`VALIDATION_RESULTS.md`, `DEV.md` and `CONTRIBUTING.md` absorbed and removed.
`validate_linear_gaussian_filter.py` dropped as agreed; `03` kept.

Corrected while absorbing: `METHODS.md` claimed the Hessian is computed by
central finite differences, but the default is now `hessian_method="autodiff"`.

The manual's headline example was run end-to-end and fits; every API name it
documents was checked to import. Tracked files: 64 → 60.

## ~~Phase 5 — Write the documentation set~~ (original plan below)

Written only after Phases 2–4 are green, so both files describe a working API
and cite measured results rather than aspirational ones.

- [ ] Rerun full suite, all `dev/` scripts, all 8 tutorials; capture real
      output and the real environment (Python, NumPy, SciPy, JAX, jaxlib)
- [ ] **`docs/VALIDATION.md`** — fold in `VALIDATION_RESULTS.md`. One section
      per `dev/` reference check: the mathematical question, the reference
      calculation, the failure criterion, the measured error. Replaces the
      stale "38 passed / all seven scripts" claim with real numbers.
- [ ] **`docs/MANUAL.md`** — the generative modelling framework: state and
      observation mappings, static vs dynamic models, priors, MAP estimation,
      the Laplace approximation, and filtering (EKF for Gaussian; Laplace-
      Gaussian with Fisher scoring for Bernoulli/categorical, labelled
      approximate). Simple math with references; **must document the logit
      contract** (Finding 3).
- [ ] Fold `DEV.md` (numerical rules, validation philosophy) and
      `CONTRIBUTING.md` into the new set; delete once absorbed
- [ ] Delete `docs/VALIDATION_RESULTS.md` once merged into `VALIDATION.md`
- [ ] Check `docs/METHODS.md` for overlap with `MANUAL.md`; merge or drop

## Phase 6 — Rename to `gbmtoolbox` — **DONE** (`c992098`)

Directory moved with `git mv` (history preserved); imports, distribution name,
`packages.find`, docstrings, the JAX cache path, the fit-summary header and the
matplotlib install hint all updated. Reinstalled editable.

**Deliberately unchanged:** the three `github.com/ginobattistello/BayesGBM`
URLs and `README.md:254`'s `cd BayesGBM`, since the repository is not being
renamed; and the inherited copyright line in `LICENSE`, which carries CBM
attribution per `NOTICE.md` — yours to decide with the author metadata.

Phase 7's README items were folded in here, since the README was the last file
still asserting the pre-JAX contract. The minimal example was extracted and
executed as printed, and produces the same estimates as the manual's worked
example. Clean-venv install verified: `gbmtoolbox` installs, pulls JAX, 49/49.

## ~~Phase 6 — Rename to `gbmtoolbox`~~ (original plan below)

`gbmtoolbox` is free on PyPI (404). Do this **after** the suite is green, so a
red result during the rename is unambiguously a rename typo.

Scope: 114 occurrences across 55 files — 60 × `bayesgbm`, 54 × `BayesGBM`.

- [ ] 44 mechanical import lines — bulk edit, then verify by import
- [ ] Directory `bayesgbm/` → `gbmtoolbox/` (`git mv` to preserve history)
- [ ] `pyproject.toml`: `name`, `packages.find.include`
- [ ] **`optimization.py:19`** — `Path.home()/".cache"/"bayesgbm"/"jax"`, a
      hardcoded cache path a blind find-and-replace would orphan
- [ ] 33 prose mentions in Markdown + `CITATION.cff` → "GBM Toolbox" by hand
      (display name, not the import name)
- [ ] `.github/workflows/tests.yml`
- [ ] **`pip install -e .` again** — the current editable install
      (`__editable__.bayesgbm-0.1.0.pth`) breaks when the directory moves
- [ ] Decide whether the GitHub repo is renamed too; if so update the
      `ginobattistello/BayesGBM` URLs in README, `pyproject.toml`,
      `CITATION.cff` — **needs your answer**
- [ ] Full suite green under the new name

## Phase 7 — Fix the stale facts

Scope deliberately limited to what is factually wrong.

- [ ] **`README.md:45` + minimal example** — correct the Bernoulli contract to
      logit (Finding 3). Highest priority item in this phase.
- [ ] `README.md` minimal example — JAX idiom, so it is copy-pasteable
- [ ] Add a short "Requires JAX" note to Installation
- [ ] `CHANGELOG.md` — add the JAX backend and the contract change; the
      current 0.1.0 entry describes the NumPy design
- [ ] Fold the still-relevant `RELEASE_CHECKLIST.md` items (author metadata,
      Zenodo DOI, tag after API freeze) into wherever you want them to live
- [ ] Replace `BayesGBM contributors` placeholder in `pyproject.toml` and
      `CITATION.cff` — **needs your final author metadata**

## Phase 8 — Verify

- [ ] Clean venv, `pip install -e ".[dev]"`, full suite green
- [ ] All 10 `dev/` scripts pass
- [ ] All 8 tutorials pass
- [ ] README minimal example runs verbatim as printed
- [ ] CI green across 3.10–3.13 — first meaningful run, since `jax` is only now
      declared

---

## Known gaps, accepted

**The README describes a NumPy-era API in prose throughout**, not only in the
example Phase 7 fixes. Under the minimal-docs scope, the surrounding narrative
is not reconciled with a JAX-only contract. Deliberate, not an oversight.

**Done 2026-09-16:** `docs/DESIGN_PLAN_v3.md` and `REPOSITORY_SETUP.md` removed
from the published tree (both recoverable in git history); local build
artifacts cleared; `.claude/` gitignored. The overlap between `CONTRIBUTING.md`,
`DEV.md` and `docs/VALIDATION.md` is resolved by the Phase 5 rewrite rather
than by patching the three files.

## Open questions

1. Is the GitHub repository being renamed alongside the package? (Phase 6)
2. Final author metadata for `CITATION.cff` and `pyproject.toml`? (Phase 7)
3. Minimum `jax` version to support, or match the 0.11.0 dev box? (Phase 1)
