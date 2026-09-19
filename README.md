# Minimal Neural Network Optimization

A nine-parameter **2→2→1** NumPy network learning XOR, with hand-written backpropagation, GD, heavy-ball momentum and Adam. No deep-learning framework is required. This is a small teaching model, not a claim of absolute minimality across all possible architectures.

从 [中文学习说明](reports/LEARNING_GUIDE.pdf) 开始，了解推导、实现顺序和真实卡点；再打开 [experiment.ipynb](experiment.ipynb) 逐步重跑。可编辑说明在 [LEARNING_GUIDE.md](reports/LEARNING_GUIDE.md)。

## Run

Use Python 3.12 from this directory, preferably in a virtual environment.

```powershell
python -m pip install -r requirements.txt
python check_gradients.py
python -m pytest -q test_model.py
python run_experiment.py
python -m jupyterlab experiment.ipynb
```

## What to read

| File | Purpose |
|---|---|
| [model.py](model.py) | Nine parameters, stable BCE, explicit chain-rule gradients |
| [check_gradients.py](check_gradients.py) | Independent central finite differences |
| [train.py](train.py) | Three optimizers from their recurrence equations |
| [run_experiment.py](run_experiment.py) | 20 paired starts, all failures retained, stress cases and plots |
| [test_model.py](test_model.py) | Derivatives, symmetry, stability and the predeclared training target |
| [results/](results/) | Model parameters, predictions, learning curves and complete run tables |

The predeclared Adam seed-0 example reaches BCE 0.0199167 and 4/4 correct truth-table entries in 194 updates. At the fixed learning rates used here, the 20-seed success counts are GD 12, momentum 12 and Adam 10. These settings are not an exhaustive optimizer benchmark; iteration medians are conditional on each method's own successful subset.

Zero initialization is an exact stationary trap. Saturated hidden units and a very large step also produce retained failures. Four-point training accuracy is not held-out generalization, and the loss target is not a proof of a finite-parameter global minimum.

`python build_documents.py` refreshes the Markdown guide and notebook from saved results; rerun the notebook afterward to restore outputs. Code and explanations were AI-assisted and independently reviewed. `work/` contains local scratch and review records and is excluded from Git. This repository is independent of the budget-allocation course project.
