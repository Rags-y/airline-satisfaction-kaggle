# Predicting Airline Satisfaction 

A machine learning project for predicting airline passenger satisfaction using classification models, feature experimentation, and ensemble learning.


## Results

| Metric | Result |
|---|---:|
| Best Public ROC AUC | **0.95837** |
| Best Submission | Day 20 – Three-Model Ensemble |
| Successful Kaggle Submissions | 10 |
| Evaluation Metric | ROC AUC |

The best-performing solution combines CatBoost, XGBoost, and LightGBM predictions.

**Best ensemble configuration:**

| Model | Weight |
|---|---:|
| CatBoost | 25% |
| XGBoost | 60% |
| LightGBM | 15% |

- Local validation ROC AUC: **0.958093**
- Kaggle public ROC AUC: **0.95837**

The final Day 21 ensemble achieved a public ROC AUC of 0.95835, slightly below the Day 20 result.


## Project Overview

The objective of this competition was to predict whether an airline passenger is satisfied with their flight experience.

The task is a binary classification problem. Models generate a probability of satisfaction for each passenger in the test dataset.

Submissions are evaluated using the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

The project explores multiple machine learning approaches, compares model performance, experiments with hyperparameters and feature engineering, and combines complementary models through weighted probability averaging.


## Dataset

The competition dataset is synthetically generated from real-world data.

The training and test files contain passenger information and flight experience attributes.

| Dataset | Rows |
|---|---:|
| Training data | 699,635 |
| Test data | 299,844 |

### Features

The dataset includes information such as:

- Passenger age and gender
- Customer type
- Type of travel
- Travel class
- Flight distance
- Inflight Wi-Fi service
- Departure and arrival convenience
- Food and drink
- Seat comfort
- Inflight entertainment
- On-board service
- Leg room service
- Baggage handling
- Check-in service
- Cleanliness
- Departure and arrival delays

**Target:** `satisfaction`

The `id` column is retained for submission generation and excluded from model training.


## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Gradient Boosting:** CatBoost, XGBoost, LightGBM
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **Development:** Jupyter Notebook, Python scripts
- **Version Control:** Git and GitHub


## Workflow and Experiments

### 1. Exploratory Data Analysis

- Inspected dataset dimensions, feature types, and distributions.
- Examined target distribution and missing values.
- Explored relationships between passenger attributes and satisfaction.

### 2. Baseline Model

Started with Logistic Regression using a preprocessing pipeline:

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- Standard scaling for numerical features.
- One-hot encoding for categorical features.

**Baseline validation ROC AUC:** 0.92046

### 3. CatBoost Experiments

Used CatBoost as the first major gradient-boosting model.

Experimented with:

- Tree depth
- Learning rate
- Random seed
- L2 regularization
- Random strength
- Feature engineering

The initial CatBoost model achieved a Kaggle public ROC AUC of 0.95821.

### 4. Feature Engineering

Explored additional features, including:

- Service rating aggregates
- Age groups
- Combined delay information
- Arrival and departure delay differences
- Missing-value indicators

The experiments helped evaluate whether additional features improved model performance.

### 5. Model Comparison

Compared three gradient-boosting approaches:

- CatBoost
- XGBoost
- LightGBM

XGBoost achieved the strongest individual validation performance among the models evaluated in the Day 12 comparison.

### 6. Ensemble Learning

Tested different weighted combinations of model predictions.

The best-performing configuration was:

- CatBoost: 25%
- XGBoost: 60%
- LightGBM: 15%

The final prediction is generated through weighted probability averaging.


## Kaggle Submission Results

All 10 submissions were successfully accepted by Kaggle.

| Experiment | Approach | Public ROC AUC |
|---|---|---:|
| Day 3 | Logistic Regression baseline | 0.92229 |
| Day 4 | CatBoost baseline | 0.95821 |
| Day 6 | CatBoost with feature engineering | 0.95817 |
| Day 7 | CatBoost seed experiment | 0.95816 |
| Day 8 | CatBoost depth experiment | 0.95805 |
| Day 9 | CatBoost learning-rate experiment | 0.95828 |
| Day 10 | CatBoost L2 regularization | 0.95824 |
| Day 17 | CatBoost + XGBoost ensemble | 0.95831 |
| Day 20 | CatBoost + XGBoost + LightGBM ensemble | **0.95837** |
| Day 21 | Final three-model weight experiment | 0.95835 |


## Project Structure

```text
airline-satisfaction-kaggle/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── models/
│   └── Trained model artifacts
│
├── outputs/
│   └── Generated Kaggle submission files
│
├── reports/
│   ├── Model evaluation metrics
│   ├── Feature importance reports
│   └── Ensemble comparison reports
│
├── day1_explore.py
├── baseline_model.py
├── day4_catboost.py
├── day5_tune_catboost.py
├── day6_feature_engineering.py
├── day7_seed_experiment.py
├── day8_depth_experiment.py
├── day9_learning_rate_experiment.py
├── day10_l2_experiment.py
├── day11_random_strength.py
├── day12_model_comparison.py
├── day13_xgboost_tuning.py
├── day14_ensemble.py
├── day15_three_model_ensemble.py
├── day16_xgboost_engineering.py
├── day18_ensemble_comparison.py
├── day19_ensemble.py
├── day20_ensemble.py
├── day21_ensemble.py
├── day21_compare_candidates.py
│
├── create_submission.py
├── create_catboost_submission.py
├── create_day6_submission.py
├── create_day7_submission.py
├── create_day8_submission.py
├── create_day9_submission.py
├── create_day10_submission.py
├── create_day17_ensemble_submission.py
├── create_day20_submission.py
├── create_day21_submission.py
│
├── requirements.txt
└── README.md
```


## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rags-y/airline-satisfaction-kaggle.git
cd airline-satisfaction-kaggle
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The comparison and ensemble experiments also use XGBoost and LightGBM:

```bash
pip install xgboost lightgbm
```

### 4. Prepare the dataset

Download the competition data from the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s6e10/data).

Place the following files inside the `data/` directory:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 5. Run experiments

For example, to run the baseline:

```bash
python baseline_model.py
```

To run the CatBoost experiment:

```bash
python day4_catboost.py
```

To run the model comparison:

```bash
python day12_model_comparison.py
```

To run the final ensemble experiment:

```bash
python day20_ensemble.py
```


## Submission Format

Kaggle expects a CSV containing an `id` column and a predicted probability in the `satisfaction` column.

```csv
id,satisfaction
699635,0.247967
699636,0.891012
699637,0.906105
```

The generated submission files are saved in the `outputs/` directory.


## Key Learnings

- Building an end-to-end binary classification workflow.
- Performing exploratory data analysis and preprocessing.
- Training and evaluating gradient-boosting models.
- Experimenting with hyperparameters and feature engineering.
- Comparing validation performance across models.
- Combining model predictions using weighted ensembles.
- Generating and validating Kaggle submission files.
- Tracking experiments and maintaining reproducible code with Git.

## Author
Raghav Kumar
