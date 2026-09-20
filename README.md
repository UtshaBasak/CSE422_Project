# Loan Approval Prediction

A supervised machine learning project that predicts the outcome of a personal loan
application from applicant demographics, financial history, and loan terms.

Three classifiers — **Logistic Regression**, **K-Nearest Neighbours**, and a
**Multi-Layer Perceptron neural network** — are trained on the same preprocessed
data and compared using ROC AUC, confusion matrices, and per-class
precision / recall / F1 scores.

> Course project for **CSE422 — Artificial Intelligence**, BRAC University.

---

## Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Notes](#notes)
- [Author](#author)

---

## Dataset

`data/Loan Approval Dataset.csv` — **45,000 records**, **14 columns**, no missing values.

### Features

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `person_age` | Numeric | Age of the applicant in years |
| 2 | `person_gender` | Categorical | `male`, `female` |
| 3 | `person_education` | Categorical | `High School`, `Associate`, `Bachelor`, `Master`, `Doctorate` |
| 4 | `person_income` | Numeric | Annual income of the applicant |
| 5 | `person_emp_exp` | Numeric | Years of employment experience |
| 6 | `person_home_ownership` | Categorical | `RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| 7 | `loan_amnt` | Numeric | Requested loan amount |
| 8 | `loan_intent` | Categorical | `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT` |
| 9 | `loan_int_rate` | Numeric | Interest rate offered on the loan |
| 10 | `loan_percent_income` | Numeric | Loan amount as a fraction of annual income |
| 11 | `cb_person_cred_hist_length` | Numeric | Length of credit history in years |
| 12 | `credit_score` | Numeric | Credit score of the applicant |
| 13 | `previous_loan_defaults_on_file` | Categorical | `Yes`, `No` |
| 14 | `loan_status` | **Target** | Binary loan outcome |

### Target distribution

| `loan_status` | Count | Share |
|---------------|-------|-------|
| `0` | 35,000 | 77.8 % |
| `1` | 10,000 | 22.2 % |

The target is **imbalanced at roughly 3.5 : 1**, which is why the comparison relies on
ROC AUC and per-class precision / recall rather than raw accuracy — a model that
predicts the majority class for every applicant would already score about 78 %.

---

## Methodology

1. **Load and clean** — read the CSV and drop rows with null values (the dataset
   contains none, so all 45,000 records are retained).
2. **Inspect categories** — print the value counts of every non-numeric column to
   confirm the full set of levels before encoding.
3. **Encode categoricals** — map each categorical column to integer codes so the
   whole frame becomes numeric:

   | Column | Mapping |
   |--------|---------|
   | `person_gender` | male → 0, female → 1 |
   | `person_education` | Bachelor → 0, Associate → 1, High School → 2, Master → 3, Doctorate → 4 |
   | `person_home_ownership` | RENT → 0, MORTGAGE → 1, OWN → 2, OTHER → 3 |
   | `loan_intent` | EDUCATION → 0, MEDICAL → 1, VENTURE → 2, PERSONAL → 3, DEBTCONSOLIDATION → 4, HOMEIMPROVEMENT → 5 |
   | `previous_loan_defaults_on_file` | Yes → 0, No → 1 |

4. **Correlation analysis** — compute the correlation matrix over the now fully
   numeric frame and render it as a `YlGnBu` heatmap.
5. **Class balance check** — count and plot the two `loan_status` classes.
6. **Split** — 80 % training / 20 % testing via `train_test_split`
   (`test_size=0.2`, `random_state=67` for reproducibility).
7. **Scale** — standardize the features with `StandardScaler` (mean 0, standard
   deviation 1), fitted on the training split only and then applied to the test
   split, so no test-set statistics leak into training.
8. **Train and compare** — fit all three models on the scaled training data and
   evaluate them on the held-out test set.

---

## Models

| Model | Configuration |
|-------|---------------|
| Logistic Regression | scikit-learn defaults |
| K-Nearest Neighbours | scikit-learn defaults (`k = 5`) |
| Neural Network (MLP) | 3 hidden layers of 8 units, ReLU activation, Adam solver, `max_iter = 1000` |

---

## Evaluation

Running the script produces the following, in order:

1. **Correlation heatmap** across all 14 columns.
2. **Class distribution bar chart** for `loan_status`.
3. **Overlaid ROC curves** for all three models on one axis, each labelled with its
   AUC, plotted against the random-guess diagonal.
4. **ROC AUC bar chart** comparing the three models side by side.
5. **Confusion matrix** for each model, plus its accuracy score and a full
   `classification_report` with per-class precision, recall, and F1.

All figures are rendered with `plt.show()` rather than written to disk, so they
appear inline in the notebook or in a plotting window when run locally.

---

## Project Structure

```
CSE422_Project/
├── data/
│   └── Loan Approval Dataset.csv    # 45,000-record source dataset
├── src/
│   └── model.py                     # full pipeline: preprocessing → training → evaluation
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### Option A — Google Colab (recommended)

`src/model.py` was authored in Colab and reads the dataset from `/content/`.
In a Colab notebook:

```python
!git clone https://github.com/UtshaBasak/CSE422_Project.git
!cp "CSE422_Project/data/Loan Approval Dataset.csv" /content/
%run CSE422_Project/src/model.py
```

Alternatively, upload `Loan Approval Dataset.csv` directly to the Colab session's
`/content/` directory and paste the cells from `src/model.py` into the notebook.
The `#@title` comments in the script are Colab cell titles and mark the boundary of
each step.

### Option B — Run locally

```bash
git clone https://github.com/UtshaBasak/CSE422_Project.git
cd CSE422_Project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python src/model.py
```

The script loads the dataset from the absolute Colab path `/content/Loan Approval
Dataset.csv`. To run it outside Colab, make the file available at that path — for
example by copying it there — or point the `pd.read_csv(...)` call in
`src/model.py` at `data/Loan Approval Dataset.csv` instead.

### Requirements

Python 3.9+ and the packages listed in [`requirements.txt`](requirements.txt):
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

---

## Notes

- **Reproducibility.** The train/test split is seeded with `random_state=67`, so the
  split is identical between runs. The `MLPClassifier` is *not* seeded, so its
  weight initialization differs from run to run and its scores will vary slightly.
  The neural network is also instantiated twice — once inside the ROC comparison
  loop and again for the confusion matrix — so those two sets of numbers come from
  two independently trained networks.
- **Label convention.** The source dataset documents `loan_status = 1` as an
  approved loan and `0` as rejected. The plot labels and the `target_names`
  arguments in `src/model.py` use the opposite convention, so read the axis labels
  of the generated charts with that in mind.
- **Ordinal encoding.** Nominal columns such as `loan_intent` and
  `person_home_ownership` are mapped to integers, which implies an ordering that
  does not exist in the data. This suits the tree-free, distance-based and linear
  models used here only after standardization; one-hot encoding would be the
  stricter choice for a follow-up.

---

## Author

**Utsha Basak**
CSE422 — Artificial Intelligence, BRAC University
