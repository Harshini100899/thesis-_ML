# Multi-Label Classification of Software Repositories Based on Dependencies

A machine learning pipeline that automatically categorizes software repositories into multiple scientific and technical domains based on their software dependencies (e.g., NumPy, pandas, TensorFlow).

---

## Requirements

### Python Version
- Python 3.8 or higher

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm catboost
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
```

---

## Input Files Required

Place these two CSV files in the **same directory** as `multilabel_classifier.py`:

| File | Description |
|------|-------------|
| `joss_all_with_dependency_labels1.csv` | Main dataset: contains `dependecies_found` and `dependency_labels` columns |
| `dependency_counts_with_labels.csv` | Lookup table: contains `dependency` and `label` columns (valid vocabulary) |

### Expected Column Names

**`joss_all_with_dependency_labels1.csv`** must contain:
- `dependecies_found` or `dependencies_found` — list of software dependencies (as string, e.g. `"['numpy', 'pandas']"`)
- `dependency_labels` — list of domain labels (e.g. `"['Bioinformatics', 'Python']"`)

**`dependency_counts_with_labels.csv`** must contain:
- `dependency` — dependency name
- `label` — corresponding domain label

---

## How to Run

```bash
python multilabel_classifier.py
```

---

## Output Files Generated

| File | Description |
|------|-------------|
| `best_model_<timestamp>.joblib` | Saved best model with preprocessors |
| `results_<timestamp>.csv` | All model performance metrics |
| `results_<timestamp>.json` | Full results including dataset info and best parameters |
| `results_report_<timestamp>.txt` | Human-readable report with recommendations |
| `model_evaluation_<timestamp>.png` | Evaluation visualizations |
| `eda_analysis_<timestamp>.png` | EDA visualizations |
| `training_validation_loss_<timestamp>.png` | Loss curves (MLP only) |

---

## Pipeline Steps

1. **Data Loading** — Loads and cleans the main dataset and lookup vocabulary
2. **Preprocessing** — Parses, normalizes, and filters dependencies and labels
3. **Feature Extraction** — Converts dependency lists to binary feature matrix using `MultiLabelBinarizer`
4. **EDA** — Visualizes label distributions and class imbalance
5. **Train/Test Split** — 75% training / 25% testing
6. **Nested Cross-Validation** — 3 outer folds × 2 inner folds for unbiased performance estimation
7. **Hyperparameter Tuning** — `GridSearchCV` on full training set
8. **Final Evaluation** — Trains and evaluates all models on held-out test set
9. **Visualization** — Generates comparison plots
10. **Save Results & Best Model** — Persists results and trained model

---

## Models Evaluated

| Model | Notes |
|-------|-------|
| Random Forest | Tree-based ensemble with balanced class weights |
| Logistic Regression (OVR) | Linear baseline, efficient on sparse data |
| Linear SVC (OVR) | Maximum margin classifier |
| MLP Classifier | Neural network with early stopping |
| XGBoost (OVR) | Gradient boosting |
| LightGBM (OVR) | Fast gradient boosting |
| CatBoost (OVR) | Gradient boosting with auto class balancing |

---

## Configuration

Edit the constants near the top of `main()` in `multilabel_classifier.py`:

```python
MAIN_FILE = 'joss_all_with_dependency_labels1.csv'
LOOKUP_FILE = 'dependency_counts_with_labels.csv'
TEST_SIZE = 0.25             # Fraction of data for testing
RANDOM_STATE = 42            # Reproducibility seed
APPLY_SCALING = False        # Apply StandardScaler to features
FEATURE_SELECTION_K = None   # Select top-K features (None = use all)
USE_NESTED_CV = True         # Run nested cross-validation (recommended)
NESTED_OUTER_FOLDS = 3       # Outer CV folds
NESTED_INNER_FOLDS = 2       # Inner CV folds (hyperparameter tuning)
```

> **Note:** Setting `USE_NESTED_CV = True` is recommended for unbiased performance estimates but may take several minutes depending on dataset size.

---

## Using the Saved Model for Inference

```python
import joblib

bundle = joblib.load('best_model_<timestamp>.joblib')
clf = bundle['model']
mlb_X = bundle['mlb_X']
mlb_y = bundle['mlb_y']
valid_deps = bundle['valid_dependencies']

# Filter and transform input dependencies
input_deps = ['numpy', 'pandas', 'scikit-learn']
filtered = [d.lower() for d in input_deps if d.lower() in valid_deps]
X_input = mlb_X.transform([filtered])

# Predict
y_pred = clf.predict(X_input)
labels = mlb_y.inverse_transform(y_pred)
print(labels)
```
