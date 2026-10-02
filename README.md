# Credit Card Fraud Detection
 
Detecting fraudulent credit card transactions in a highly imbalanced dataset (~0.58% fraud) and comparing **anomaly detection** with **supervised learning**.
 
## Overview
 
The project follows three steps, each in its own notebook:
 
| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Explore the data: distributions, outliers, class imbalance, correlations |
| `02_data_cleaning.ipynb` | Remove noise, engineer features (hour, day of week, distance, age), encode categories |
| `03_model_training.ipynb` | Train, tune, and compare the models |
 
## Dataset
 
[Credit Card Transactions Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
 
* `fraudTrain.csv`: about 1.3 million transactions, of which 0.579% are fraud.
* The data is simulated, so results may not transfer directly to real-world data.
The dataset is downloaded automatically in the notebooks using `kagglehub`.
 
## Feature Engineering
 
* **Dropped:** personal and ID columns (names, card number, street, job, zip, transaction number) to protect privacy and avoid memorization.
* **Time:** `hour` and `day of week` extracted from the transaction timestamp.
* **Distance:** `distance_km` between the customer and the merchant, calculated from coordinates.
* **Age:** calculated from date of birth.
* **Encoding:** frequency encoding for `merchant` and `city`, one-hot encoding for `state` and `category`.
Final cleaned dataset: 1,296,675 rows and 77 columns.
 
## Models
 
1. **Isolation Forest** (anomaly detection): learns what normal transactions look like and flags outliers. It is fitted without labels. The labels are used only to set `contamination` and to evaluate the results.
2. **HistGradientBoosting** (supervised, `class_weight='balanced'`): used as a benchmark to show how much the labels help.
Data split: 80% train / 20% test, stratified, `random_state=42`.
 
Accuracy is not used as the main metric, because a model that always predicts "no fraud" would already reach about 99.4%. The evaluation uses **Precision, Recall, F1, and PR-AUC**.
 
## Results
 
| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.128 | 0.125 | 0.126 | 0.066 |
| HistGradientBoosting (threshold 0.5) | 0.321 | 0.974 | 0.482 | 0.925 |
| HistGradientBoosting (tuned threshold 0.974) | 0.930 | 0.824 | 0.874 | 0.925 |
 
HistGradientBoosting also reached ROC-AUC = 0.998.
 
### Key findings
 
* **Isolation Forest performed close to random** (PR-AUC 0.066, the baseline is 0.0058). Fraudulent transactions are not strong statistical outliers in this dataset, and `distance_km` alone does not separate fraud from normal transactions.
* **HistGradientBoosting performed far better.** The default threshold catches 97% of fraud but raises many false alarms. Tuning the threshold keeps recall at 82% while raising precision from 0.32 to 0.93.
* In a real bank, missing a fraud usually costs more than a false alarm, so the threshold can be moved toward higher recall depending on business needs.
## Limitations
 
* The decision threshold was selected on the same test set used for evaluation, so the tuned numbers may be slightly optimistic. PR-AUC and ROC-AUC do not depend on the threshold.
* The split is random, so transactions of the same customer may appear in both train and test, which may inflate the results.
* The dataset is simulated.
## Possible Next Steps
 
* Evaluate on `fraudTest.csv` (using the same cleaning steps) for a later time period.
* Split the data by customer to test generalization to unseen customers.
* Add behavioral features, such as the transaction amount relative to each customer's average.
## How to Run
 
The notebooks were developed in **Google Colab**.
 
1. Run `01_data_exploration.ipynb` (optional, exploration only).
2. Run `02_data_cleaning.ipynb`. It saves `final_cleaned_fraud_data.csv`. The distance calculation takes a few minutes on 1.3M rows.
3. Run `03_model_training.ipynb`. It reads the cleaned file (from Google Drive in the Colab version) and trains both models.
**Requirements:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `kagglehub`, `geopy`, `joblib`.
 
## Files
 
| File | Description |
|---|---|
| `fraud_model.joblib` | Trained HistGradientBoosting model and its tuned decision threshold (0.974) |
| `sample_cleaned_data.csv` | Small demo sample: 5,000 rows, 76 features + `is_fraud` target |
 
> **Note about the sample:** it contains 10% fraud (500 fraud + 4,500 normal), **not** the original 0.58% ratio. It is meant for demonstration only, not for evaluating the model.
 
The full cleaned dataset (1.3M rows) is not included because of its size. Run `02_data_cleaning.ipynb` to generate it.
 
## Load the Model
 
```python
import joblib
import pandas as pd
 
bundle = joblib.load('fraud_model.joblib')
model = bundle['model']
threshold = bundle['threshold']
 
sample = pd.read_csv('sample_cleaned_data.csv')
X = sample.drop(columns=['is_fraud']).astype(float)
 
proba = model.predict_proba(X)[:, 1]
pred = (proba >= threshold).astype(int)
 
print('Flagged as fraud:', pred.sum(), 'of', len(sample))
```
 
Expected output on the demo sample: `Flagged as fraud: 401 of 5000`.
 

 
