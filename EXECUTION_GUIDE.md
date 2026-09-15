# VS Code Execution Guide - AI-Driven Hostel Food Waste Mitigation and Demand Forecasting

This guide assumes Windows, VS Code, Git integration, and Python are already installed.

## 1. Open the project correctly
1. Extract `hostel-food-waste-ai-capstone.zip`.
2. In VS Code select **File > Open Folder**.
3. Select the folder `hostel-food-waste-ai` (the folder containing `README.md`, `src`, `data`, and `requirements.txt`).
4. Open **Terminal > New Terminal**.
5. Confirm the prompt is located at the repository root.

Expected structure:
```text
hostel-food-waste-ai/
  src/
  data/
  models/
  reports/
  docs/
  requirements.txt
  README.md
```

## 2. Confirm Python and Git
Run:
```powershell
python --version
git --version
```
Recommended Python: 3.10, 3.11, or 3.12.

If `python` is not recognized, try:
```powershell
py --version
```
and use `py` in place of `python` in subsequent commands.

## 3. Create an isolated virtual environment
From the repository root:
```powershell
python -m venv .venv
```
Activate it:
```powershell
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
The terminal should now show `(.venv)`.

## 4. Select the same Python interpreter in VS Code
1. Press `Ctrl+Shift+P`.
2. Search **Python: Select Interpreter**.
3. Choose `.venv\Scripts\python.exe`.

## 5. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Verify key packages:
```powershell
python -c "import pandas,numpy,sklearn,xgboost,joblib; print('Dependencies OK')"
```
Expected:
```text
Dependencies OK
```

## 6. Generate the synthetic hostel dataset
Run this as one line in PowerShell:
```powershell
python src/generate_synthetic_data.py --output data/hostel_food_waste.csv --start-date 2024-01-01 --end-date 2025-12-31 --hostel-capacity 520 --seed 42
```
Expected core result:
```text
Generated 2,193 synthetic meal records -> data/hostel_food_waste.csv
```
Open `data/hostel_food_waste.csv` in VS Code and verify columns such as date, meal category, resident/booking information, turnout, food prepared, and food wasted.

## 7. Train and evaluate all models
```powershell
python src/train_predictive_pipeline.py --data data/hostel_food_waste.csv --output-dir reports --model-dir models --train-fraction 0.80 --safety-buffer 0.03 --seed 42
```
The run should create/update:
```text
reports/model_metrics.csv
reports/holdout_predictions.csv
models/best_turnout_model.joblib
models/model_metadata.json
```
Expected approximate metrics:
```text
LinearRegression  MAE 10.158  RMSE 12.825  R2 0.9736
XGBoost           MAE 10.118  RMSE 13.044  R2 0.9727
RandomForest      MAE 10.565  RMSE 13.521  R2 0.9706
```
Small floating-point display differences are acceptable; major differences indicate changed dependencies, data, seed, or source code.

## 8. Test single-meal prediction
Inspect `examples/next_meal.json`, then run:
```powershell
python src/predict.py --input examples/next_meal.json
```
Expected output is JSON containing:
```text
predicted_turnout
recommended_food_prep_kg
safety_buffer_pct
model
```
This is the deployment-style inference demo for your capstone presentation.

## 9. Run automated tests
```powershell
pytest -q
```
Expected:
```text
2 passed
```
If XGBoost emits a warning but tests pass, distinguish a warning from an actual test failure.

## 10. Initialize/push to GitHub
If this folder is not yet a Git repository:
```powershell
git init
git add .
git commit -m "Initial AI hostel food waste forecasting capstone"
git branch -M main
```
Create an empty GitHub repository named `hostel-food-waste-ai`, then connect it:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/hostel-food-waste-ai.git
git push -u origin main
```
If a remote already exists, check it with:
```powershell
git remote -v
```

## 11. Normal development cycle
After changing code:
```powershell
python src/generate_synthetic_data.py --output data/hostel_food_waste.csv --start-date 2024-01-01 --end-date 2025-12-31 --hostel-capacity 520 --seed 42
python src/train_predictive_pipeline.py --data data/hostel_food_waste.csv --output-dir reports --model-dir models --train-fraction 0.80 --safety-buffer 0.03 --seed 42
pytest -q
git status
git add .
git commit -m "Describe the change"
git push
```

## 12. What to demonstrate during viva/interview
Demonstrate in this order:
1. Problem: uncertain hostel meal turnout creates overproduction and shortages.
2. Repository structure and reproducibility.
3. Synthetic generator and its validation rules.
4. Leakage boundary: post-meal quantities are not turnout predictors.
5. Chronological 80/20 split.
6. Linear Regression vs Random Forest vs XGBoost.
7. MAE, RMSE, and R2 results.
8. Why Linear Regression wins RMSE on this synthetic generator while XGBoost wins MAE.
9. Conversion of turnout to kilograms using training-only meal-specific consumption.
10. 3% configurable safety buffer.
11. Single-meal `predict.py` demo.
12. GitHub commit history and automated tests.
13. Limitation: synthetic results are engineering validation, not measured hostel savings.
14. Next phase: audited real data, FastAPI/dashboard, uncertainty/quantile forecasting, drift monitoring.

## 13. Troubleshooting
### `python` not found
Try `py` or reinstall/select Python in VS Code.

### `ModuleNotFoundError`
Confirm `(.venv)` appears, then run:
```powershell
pip install -r requirements.txt
```

### PowerShell activation denied
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Wrong paths/file not found
Run:
```powershell
pwd
Get-ChildItem
```
You must be in the folder containing `src` and `requirements.txt`.

### Git remote already exists
```powershell
git remote -v
git remote set-url origin https://github.com/YOUR_USERNAME/hostel-food-waste-ai.git
```

### Reproducibility mismatch
Check Python/package versions, confirm seed `42`, do not manually edit the generated CSV, and regenerate before retraining.
