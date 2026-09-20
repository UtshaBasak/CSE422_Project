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
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Notes](#notes)
- [Author](#author)

---

## Dataset

`data/Loan Approval Dataset.csv` — **45,000 records**, **14 columns**, no missing values.

### Features

| # | Column | Type | Description |
| --- | -------- | ------ | ------------- |
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
| 14 | `loan_status` | **Target** | `1` = loan approved, `0` = loan rejected |

### Target distribution

| `loan_status` | Meaning | Count | Share |
| --------------- | --------- | ------- | ------- |
| `0` | Rejected | 35,000 | 77.8 % |
| `1` | Approved | 10,000 | 22.2 % |

The target is **imbalanced at roughly 3.5 : 1**, which is why the comparison relies on
ROC AUC and per-class precision / recall rather than raw accuracy — a model that
rejected every applicant would already score about 78 %.

One constraint is worth knowing before reading the results: every one of the
22,858 applicants with a previous default on file falls in the rejected class, and
none in the approved class. `previous_loan_defaults_on_file` therefore separates a
large part of the data on its own, and every model here leans on it heavily.

---

## Methodology

1. **Load and clean** — read the CSV and drop rows with null values (the dataset
   contains none, so all 45,000 records are retained).
2. **Inspect categories** — print the value counts of every non-numeric column to
   confirm the full set of levels before encoding.
3. **Encode categoricals** — map each categorical column to integer codes so the
   whole frame becomes numeric:

   | Column | Mapping |
   | -------- | --------- |
   | `person_gender` | male → 0, female → 1 |
   | `person_education` | Bachelor → 0, Associate → 1, High School → 2, Master → 3, Doctorate → 4 |
   | `person_home_ownership` | RENT → 0, MORTGAGE → 1, OWN → 2, OTHER → 3 |
   | `loan_intent` | EDUCATION → 0, MEDICAL → 1, VENTURE → 2, PERSONAL → 3, DEBTCONSOLIDATION → 4, HOMEIMPROVEMENT → 5 |
   | `previous_loan_defaults_on_file` | Yes → 0, No → 1 |

4. **Correlation analysis** — compute the correlation matrix over the now fully
   numeric frame and render it as a `YlGnBu` heatmap.
5. **Class balance check** — count and plot the two `loan_status` classes.
6. **Split** — 80 % training / 20 % testing via `train_test_split`
   (`test_size=0.2`, `random_state=67`), giving 36,000 training and 9,000 test records.
7. **Scale** — standardize the features with `StandardScaler` (mean 0, standard
   deviation 1), fitted on the training split only and then applied to the test
   split, so no test-set statistics leak into training.
8. **Train and compare** — fit all three models once on the scaled training data,
   then evaluate those same fitted models on the held-out test set: ROC curves and
   AUC first, then a confusion matrix and classification report for each.

---

## Models

| Model | Configuration |
| ------- | --------------- |
| Logistic Regression | scikit-learn defaults (`lbfgs` solver) |
| K-Nearest Neighbours | scikit-learn defaults (`k = 5`) |
| Neural Network (MLP) | 3 hidden layers of 8 units, ReLU activation, Adam solver, `max_iter = 1000`, `random_state = 67` |

---

## Results

Measured on the 9,000-record held-out test set (7,036 rejected / 1,964 approved).

| Model | ROC AUC | Accuracy |
| ------- | --------- | ---------- |
| **Neural Network (MLP)** | **0.967** | **0.920** |
| Logistic Regression | 0.955 | 0.900 |
| K-Nearest Neighbours | 0.936 | 0.903 |

Per-class precision / recall / F1:

| Model | Class | Precision | Recall | F1 |
| ------- | ------- | ----------- | -------- | ----- |
| Neural Network | Rejected (0) | 0.94 | 0.96 | 0.95 |
| Neural Network | Approved (1) | 0.84 | 0.78 | 0.81 |
| Logistic Regression | Rejected (0) | 0.93 | 0.94 | 0.94 |
| Logistic Regression | Approved (1) | 0.78 | 0.76 | 0.77 |
| KNN | Rejected (0) | 0.93 | 0.95 | 0.94 |
| KNN | Approved (1) | 0.80 | 0.74 | 0.77 |

The neural network wins on every measure. The gap between the two classes is the
part worth reading: all three models handle the majority *rejected* class well and
lose ground on the minority *approved* class, where recall falls to 0.74 – 0.78.
In practice that means roughly a quarter of the applicants who should be approved
are predicted as rejections — the cost of the 3.5 : 1 imbalance, and the clearest
target for future work.

The script also produces five figures, rendered with `plt.show()` rather than
written to disk: the correlation heatmap, the class distribution bar chart, the
overlaid ROC curves for all three models against the random-guess diagonal, the
ROC AUC comparison bar chart, and one confusion matrix per model.

> Produced on Python 3.13 with the pinned dependency set in
> [`requirements.txt`](requirements.txt). Both sources of randomness are seeded, so
> these numbers are identical from run to run and reproduce on a clean install.

---

## Project Structure

```
CSE422_Project/
├── .github/
│   └── dependabot.yml               # weekly dependency update checks
├── data/
│   └── Loan Approval Dataset.csv    # 45,000-record source dataset
├── src/
│   └── model.py                     # full pipeline: preprocessing → training → evaluation
├── requirements.txt                 # pinned Python dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### Run locally

```bash
git clone https://github.com/UtshaBasak/CSE422_Project.git
cd CSE422_Project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python src/model.py
```

The script resolves the dataset relative to its own location, so it runs from any
working directory without editing paths.

### Run in Google Colab

```python
!git clone https://github.com/UtshaBasak/CSE422_Project.git
%run CSE422_Project/src/model.py
```

`src/model.py` was originally authored as a Colab notebook, and the `#@title`
comments it still carries are Colab cell titles marking the boundary of each step.

### Requirements

**Python 3.12 or newer**, plus the packages in
[`requirements.txt`](requirements.txt). Versions are pinned to the exact set the
[Results](#results) were measured with, so a clean install reproduces those numbers:

| Package | Version |
| ------- | ------- |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |

The Python floor is set by numpy 2.5.3, which requires 3.12 or newer.

---

## Notes

- **Reproducibility.** Both sources of randomness are seeded with `random_state = 67`
  — the train/test split and the `MLPClassifier` weight initialization — so repeated
  runs give identical numbers. Each model is fitted once and that same fitted
  estimator is reused for both the ROC comparison and its confusion matrix, so the
  two sets of figures always describe the same model. Dependency versions are pinned
  in `requirements.txt`, so the numbers hold across machines as well as across runs.
- **Ordinal encoding.** Nominal columns such as `loan_intent` and
  `person_home_ownership` are mapped to integers, which implies an ordering that
  does not exist in the data. One-hot encoding would be the stricter choice, and is
  the natural next change to try.
- **Handling the imbalance.** Nothing in the current pipeline compensates for the
  3.5 : 1 class ratio. Class weighting, resampling, or tuning the decision threshold
  would all be reasonable ways to lift recall on the approved class.

---

## Author

**Utsha Basak**
CSE422 — Artificial Intelligence, BRAC University
