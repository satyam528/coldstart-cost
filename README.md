# Cold-Start Cost Prediction (AWS Lambda)

Predicts the **extra dollar cost caused by a cold start** of an AWS Lambda
function, using regression (SLR, MLR, Ridge, Random Forest), and serves the
best model through a small Django web app.

## Problem
When a Lambda function has been idle, AWS must boot a new container before
running it (a *cold start*). That boot time costs extra money. This project
predicts that overhead from things known **before** the call.

- **Target:** `cost_overhead_usd` = cost of the cold-start (init) part of an invocation
- **Features:** memory (MB), runtime (python / node / java), package size (MB), idle gap (min)

## Data (important)
The dataset is **simulated** (1,500 rows, `notebooks/make_data.py`). Boot times
are generated from assumed, realistic-looking ranges and converted to dollars with
the Lambda pricing formula (`GB x seconds x price per GB-second`).

**Limitation:** a model trained on simulated data can only learn the patterns put
into the generator. Results are illustrative, not measured on real AWS.
**Future work:** collect real data from CloudWatch (Init Duration / Billed Duration).

## Method
1. Generate data -> `data/coldstart_data.csv`
2. EDA (distribution, correlation heatmap)
3. Compare models with 80/20 split + 5-fold cross-validation
4. Diagnostics: VIF (multicollinearity), residual plots, actual-vs-predicted
5. Save best model -> `models/model.pkl`
6. Django app loads the model and predicts from a form

## Results (target in micro-USD)
| Model | CV R2 | Test R2 | Test RMSE | Test MAE |
|---|---|---|---|---|
| Random Forest | 0.9905 | 0.9904 | 0.948 | 0.632 |
| MLR | 0.8277 | 0.8367 | 3.905 | 3.061 |
| Ridge | 0.8277 | 0.8365 | 3.906 | 3.052 |
| MLR (log target) | 0.8153 | 0.8330 | 3.948 | 2.702 |
| SLR (memory only) | 0.4972 | 0.5229 | 6.673 | 4.859 |

VIF for all numeric features is about 1.0, so there is no multicollinearity.
Random Forest wins because cost = memory x time is a non-linear relationship that a
straight-line model cannot capture. See `reports/` for plots and the MLR summary.

## Run it
```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

cd notebooks
python make_data.py              # creates data/coldstart_data.csv
python train_model.py            # trains, saves models/model.pkl, reports/*
cd ../app
python manage.py runserver       # open http://127.0.0.1:8000
```

## Project structure
```
data/        dataset (CSV)
notebooks/   make_data.py, train_model.py
models/      model.pkl, model_info.json
reports/     plots, results.csv, vif.csv, mlr_summary.txt
app/         Django project (coldstart_site) + app (predictor)
```
