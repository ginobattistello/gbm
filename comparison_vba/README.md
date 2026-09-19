# GBM Toolbox vs. VBA Toolbox — observation-only comparison

Simulates three datasets (continuous, binary, categorical outcomes), fits each
with **both** GBM Toolbox (Python) and the VBA Toolbox (MATLAB), and compares
the parameter estimates subject by subject.

All three models are **observation-only**: no hidden-state dynamics and no
evolution parameters, so the only thing being compared is the inference
scheme — MAP/Laplace against Variational Laplace.

## Files

| file | role |
|---|---|
| `compare_gbm_vba.ipynb` | the notebook; run it top to bottom |
| `simulate.py` | generative equations for the three datasets |
| `fit_vba.m` | loops over subjects and calls `VBA_NLStateSpaceModel` |
| `build_notebook.py` | regenerates the notebook (`python build_notebook.py`) |

`sim_data.mat` and `vba_fits.mat` are produced at run time and can be deleted.

## Running

The notebook shells out to MATLAB itself, so a single top-to-bottom run does
everything. Edit the two paths in section 1 if your installs differ:

```python
MATLAB   = "/Users/gino.diez/Applications/Matlab_R2025b.app/bin/matlab"
VBA_PATH = "/Users/gino.diez/Documents/MATLAB/VBA-toolbox-master"
```

```bash
jupyter notebook compare_gbm_vba.ipynb
```

## Results (20 subjects, 200 trials, seed 1)

Correlation of the per-subject estimates across toolboxes, and of each
toolbox against the true generating values:

| model | parameter | r(GBM,VBA) | MAD | r(GBM,true) | r(VBA,true) |
|---|---|---|---|---|---|
| continuous | intercept | 1.0000 | 0.0000 | 0.9947 | 0.9947 |
| continuous | slope | 1.0000 | 0.0001 | 0.9936 | 0.9936 |
| continuous | noise SD | 1.0000 | 0.0052 | 0.9767 | 0.9764 |
| binary | b0 | 1.0000 | 0.0005 | 0.9505 | 0.9506 |
| binary | b1 | 1.0000 | 0.0011 | 0.9690 | 0.9690 |
| categorical | a1 | 1.0000 | 0.0005 | 0.9410 | 0.9411 |
| categorical | b1 | 1.0000 | 0.0025 | 0.8735 | 0.8734 |
| categorical | a2 | 1.0000 | 0.0009 | 0.8597 | 0.8595 |
| categorical | b2 | 0.9999 | 0.0037 | 0.7847 | 0.7808 |

The two toolboxes agree to 3–4 decimal places on every parameter. The largest
residual differences are in the noise SD and in `b2` — the parameters where the
likelihood is flattest and the prior does the most work. Since both toolboxes
now default to the same `Ga(1, 1)` noise prior, the noise-SD difference fell
from 0.0075 to 0.0052; the binary and categorical models are unaffected,
because they have no estimated `R`.

## Convention differences worth knowing

These are the things that silently fit the wrong model if you get them wrong:

|  | GBM Toolbox | VBA Toolbox |
|---|---|---|
| observation returns | **logits** (toolbox applies sigmoid/softmax) | **probabilities** (you apply the link) |
| categorical `y` | integer labels `0..K-1` | one-hot, `K x T` |
| null evolution | omit `evolution` and the evolution prior | `f_fname = []`, `dim.n = 0` |
| Gaussian noise | `Ga(1,1)` on precision, fitted as `log_observation_sd` | `Ga(1,1)` on the precision |
| evidence | Laplace log-evidence | variational free energy `F` |

In GBM Toolbox an observation-only model simply omits the `evolution`
function and the evolution prior; `theta` is then an empty vector.

Both toolboxes now default to the same `Ga(1, 1)` prior on observation
precision. GBM Toolbox moment-matches it onto `log_observation_sd`, because it
estimates noise inside one joint MAP/Laplace vector and the noise posterior is
much less skewed on the log scale; VBA keeps the Gamma form because its
variational scheme updates precision as a separate conjugate factor.

Note that the observation function returning logits does not hide the
probabilities: `fit.output.prediction` holds `p(y=1)` for Bernoulli and the
full per-class probability vector for categorical outcomes. Section 6 of the
notebook plots these.
