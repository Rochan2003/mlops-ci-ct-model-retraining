# CI/CT Pipeline for Model Retraining and Calibration

IE 7374 MLOps, Lab Assignment 1

Based on [Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2)
from the course repo, with the changes listed below.

A GitHub Actions pipeline that runs tests, retrains the model, calibrates it,
picks the best of four candidates, and commits the versioned model and its
metrics back into the repo.

## How it works

The workflow has two jobs. The first runs pytest. The second only starts if
those tests pass (`needs: test`), and it:

1. generates a timestamp to version this run
2. trains a Random Forest and a logistic regression baseline
3. builds two calibrated versions of the forest (sigmoid and isotonic)
4. scores all four on the held-out test set and picks a champion
5. checks the champion's F1 against a threshold
6. commits the models and metrics back to the repo

If step 5 fails, the job stops there and nothing gets committed, so a bad model
can't end up in the repo.

There are two workflows. `model_retraining_on_push.yml` runs on push to main or
manually, with the gate at F1 0.92. `model_retraining_scheduled.yml` runs weekly
and calls the same pipeline through `workflow_call` with a stricter 0.95
threshold.

## Dataset and models

Breast Cancer Wisconsin from scikit-learn. 569 samples, 30 features, binary
classification. 455 rows for training, 114 held back for testing, stratified.

Four candidates get compared each run:

- Random Forest (200 trees)
- Random Forest + sigmoid calibration
- Random Forest + isotonic calibration
- Logistic regression baseline

## Results

| Model | Accuracy | F1 | ROC-AUC | Brier |
|---|---|---|---|---|
| RF uncalibrated | 0.9561 | 0.9655 | 0.9931 | 0.0330 |
| RF + sigmoid | 0.9561 | 0.9655 | 0.9927 | 0.0338 |
| RF + isotonic | 0.9474 | 0.9583 | 0.9927 | 0.0382 |
| Logistic regression | 0.9825 | 0.9861 | 0.9954 | 0.0215 |

Champion: logistic regression, which had the lowest Brier score. Gate passed
(F1 0.9861, threshold 0.92).

Brier score is the mean squared error of the predicted probabilities, so lower
is better.

## Two things I didn't expect

**Calibration made the forest slightly worse** (Brier 0.0330 to 0.0338). I had
read that Random Forests are usually badly calibrated, so I expected an
improvement. I think this dataset is small and easy enough that the forest
already separates the classes almost perfectly (ROC-AUC 0.993), so there isn't
much miscalibration left to correct, and fitting a correction curve on only 455
rows adds noise instead.

To check that this was the data and not a bug in my code, I added a test that
deliberately under-fits a forest (50 trees, max_depth=3) so it really is
miscalibrated. On that one, sigmoid calibration improves Brier from 0.0392 to
0.0365, which is what I expected to see. So the calibration code works, this
dataset just doesn't need it.

**The logistic regression beat the Random Forest** on every metric. This is a
small clean tabular dataset, so that isn't unusual.

## What I changed from the original lab

Dataset and models:
- Switched from `make_classification()` synthetic data to the real breast
  cancer dataset
- Added a logistic regression baseline alongside the Random Forest

New code:
- `src/calibrate_model.py`, which does the Platt and isotonic calibration the
  lab README describes
- Champion selection in `evaluate_model.py`, which scores four candidates and
  picks the best by Brier score
- A quality gate that exits non-zero below a minimum F1, so the commit step
  doesn't run if the model is bad
- `tests/test_pipeline.py` with 12 tests, including the calibration sanity
  check above
- Split the workflow into a test job and a training job, so training only
  happens if the tests pass
- Made the push workflow reusable with `workflow_call` so the scheduled one
  calls it with a stricter gate instead of duplicating the steps. Added
  `workflow_dispatch` to both so I can start a run by hand

Fixes I made while getting it working:
- Moved the train/test split into `src/data.py` so training and evaluation use
  the same data, instead of generating separate random datasets
- Added a proper stratified train/test split
- Added `permissions: contents: write` to the workflow, otherwise the commit
  step fails with a 403
- Updated the actions from v2 to v4/v5
- Passed the timestamp through `$GITHUB_ENV` instead of a shell variable, since
  each workflow step gets its own shell
- Used `mlflow.set_experiment()` instead of `create_experiment()`, which errors
  if the experiment already exists
- Pinned the dependency versions

## Running it locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -v

TS=$(date '+%Y%m%d_%H%M%S')
python -m src.train_model     --timestamp "$TS"
python -m src.calibrate_model --timestamp "$TS"
python -m src.evaluate_model  --timestamp "$TS" --min-f1 0.92
```

To see the gate reject a model, set the threshold higher than anything can
reach:

```bash
python -m src.evaluate_model --timestamp "$TS" --min-f1 0.99
```

## Layout

```
.github/workflows/model_retraining_on_push.yml     main pipeline (push / manual)
.github/workflows/model_retraining_scheduled.yml   weekly run, reuses the pipeline
src/data.py                                        dataset and train/test split
src/train_model.py                                 trains RF and the baseline
src/calibrate_model.py                             sigmoid and isotonic calibration
src/evaluate_model.py                              scoring, champion selection, gate
tests/test_pipeline.py                             tests
models/                                            versioned models, written by CI
metrics/                                           versioned metrics, written by CI
```

Everything is stamped with the run timestamp so runs don't overwrite each
other.

## Limitations

- MLflow logs to a local directory, which gets destroyed with the CI runner.
  It's uploaded as a build artifact, but a real setup would need a tracking
  server.
- The gate threshold is hardcoded. It would be better to compare against the
  previous champion's score instead of a fixed number.
- The dataset is static, so scheduled retraining produces the same model every
  time.
