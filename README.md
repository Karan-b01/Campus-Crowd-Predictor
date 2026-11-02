# Campus Crowd Prediction

Project foundation for a reproducible, chronological machine learning study of
campus gym occupancy. Dataset-specific analysis and modeling will be added after
the raw CSV is placed in `data/raw/` and its contents have been verified.

## Setup

From this project directory, run:

```sh
python -m venv .venv
```

Activate the environment and install the dependencies:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

The same environment setup and dependency installation can be run with `make setup`.

## Why reproducibility matters

- A fixed seed makes randomized steps repeatable, so reruns are easier to compare.
- Chronological splits preserve the order of observations and avoid future-to-past leakage.
- Minimum dependency versions reduce surprises from incompatible library updates.
- Repeatable results make it easier to find bugs and explain choices in a viva.
- Saved configuration and code let another student reproduce the same experiment.
- Reproducibility helps separate real improvements from random variation.
