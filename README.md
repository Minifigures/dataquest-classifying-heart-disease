# Classifying Heart Disease with Logistic Regression

Binary classification on the Cleveland heart disease dataset (303 patients, 13 predictors), treating the coefficients as the deliverable rather than a by-product.

Dataquest guided project, Data Scientist in Python path.

Two investigations make up most of the notebook: whether the codebook or the data should decide how the integer-coded categoricals are encoded, and whether a single 61-patient test split can rank three models. The answers are no and no.

## The model

L2-penalised logistic regression, `C=0.1`, chosen from 18 configurations by cross-validated ROC AUC (0.9097). L1 was tested and not selected.

| Metric | Test (61 patients) |
|---|---|
| Accuracy | 0.8852 |
| Precision | 0.8621 |
| Recall | 0.8929 |
| F1 | 0.8772 |
| ROC AUC | 0.9502 |
| Majority-class baseline accuracy | 0.5410 |

Confusion matrix: 29 TN, 4 FP, 3 FN, 25 TP.

## What the coefficients say

Per standard deviation, sorted by magnitude:

| Feature | Coefficient | Odds ratio |
|---|---|---|
| `ca` (vessels on fluoroscopy) | +0.7016 | 2.02 |
| `thal` (thalassemia scan) | +0.5470 | 1.73 |
| `cp_4` (asymptomatic chest pain) | +0.4809 | 1.62 |
| `sex` (male) | +0.4614 | 1.59 |
| `slope_2` (flat ST segment) | +0.3393 | 1.40 |
| `thalach` (max heart rate) | -0.2831 | 0.75 |
| `exang` (exercise angina) | +0.2645 | 1.30 |
| `oldpeak` (ST depression) | +0.2046 | 1.23 |
| `age` | +0.0219 | 1.02 |

`age` carries essentially no coefficient. That is not evidence that age does not matter clinically, it is evidence that what age contributes is already captured by the exercise-test variables downstream of it. A univariate model would show age as strongly predictive. The gap between marginal and conditional association is the standard reason not to read regression coefficients as a ranking of clinical importance.

## Encoding: the codebook was wrong twice

Four predictors are nominal categories stored as integers. The usual move is to one-hot encode whatever the codebook calls nominal. Checking the data first:

| Column | Codebook says | Disease rate across levels | Monotonic? |
|---|---|---|---|
| `cp` | nominal | 0.304, 0.180, 0.209, 0.729 | no |
| `restecg` | nominal | 0.371, 0.750, 0.541 | no |
| `slope` | **ordinal** | 0.254, 0.650, 0.571 | **no** |
| `thal` | **nominal** | 0.223, 0.667, 0.761 | **yes** |
| `ca` | ordinal count | 0.261, 0.677, 0.816, 0.850 | yes |

`slope` has a real ordering (upsloping, flat, downsloping) but risk is not ordered along it: flat is riskier than downsloping, so one coefficient cannot represent it. `thal` is clinically unordered yet its risk rises steadily across the arbitrary codes 3, 6 and 7. The codebook reading gets both backwards.

All four encodings, fitted on identical folds (5-fold CV repeated 6 times):

| Treatment | Features | CV ROC AUC | std |
|---|---|---|---|
| A: all coded columns numeric | 13 | 0.9054 | 0.0361 |
| B: codebook nominal | 17 | 0.9074 | 0.0320 |
| **C: empirically non-monotonic** | 17 | **0.9090** | 0.0332 |
| D: every multi-level coded column | 18 | 0.9084 | 0.0328 |

The data-driven encoding wins, and **the spread between best and worst is 0.0036 against a fold-to-fold standard deviation of 0.0335**, nine times smaller than the noise. On 303 rows this decision does not move the metric, and calling treatment C an improvement would be reporting noise.

It still matters, for a different reason: the coefficients are the deliverable, and a coefficient on a non-monotonic column is not interpretable whatever it does to AUC. The encoding was chosen for readability, not performance.

## Why a single test split should not be trusted

On one 61-patient split, logistic regression looked like the worst of three models:

| Model | CV ROC AUC | Test accuracy | Test recall | Test ROC AUC | Missed cases |
|---|---|---|---|---|---|
| Logistic regression (tuned) | 0.9097 | 0.8852 | 0.8929 | 0.9502 | 3 |
| KNN (k=15, distance) | 0.8829 | 0.9016 | 0.9286 | 0.9610 | 2 |
| Random forest | 0.8918 | 0.9016 | 0.9643 | 0.9524 | 1 |

The CV ranking and the test ranking are exactly reversed. Repeating the whole evaluation across 25 random stratified splits:

| Model | Mean test AUC | std | min | max |
|---|---|---|---|---|
| **Logistic regression** | **0.9116** | **0.0319** | 0.8366 | 0.9589 |
| KNN (k=15, distance) | 0.8917 | 0.0363 | 0.8090 | 0.9610 |
| Random forest | 0.9000 | 0.0356 | 0.8149 | 0.9594 |

Logistic regression is in fact the best of the three on average and the most stable. Each model's own range spans 0.12 to 0.15 AUC, six to eight times the largest gap between models. The single-split ranking was noise, wrong in both ordering and confidence.

At n=303 a single 20% holdout cannot distinguish these models, and the linear one is competitive with both non-linear alternatives while being the only one whose decision process can be inspected.

## Method

1. **Load correctly**: `index_col=0` to drop the R row index, `na_values='?'` so the six missing values become `NaN` instead of forcing object dtype.
2. **EDA** on continuous predictors by outcome and disease rate by level for every coded column.
3. **Measure monotonicity** per coded column rather than trusting the codebook.
4. **Build four encodings** and compare on identical repeated folds.
5. **Leak-free `Pipeline`** of median imputer, standard scaler and estimator, refitted per fold.
6. **Tune the regularisation** across L1 and L2 and nine values of `C`, because the default `C=1` silently shrinks the coefficients that the notebook then interprets.
7. **Read the coefficients** as odds ratios, with the standardisation caveat stated.
8. **Compare** against KNN and a random forest, then **repeat across 25 splits** when the rankings disagreed.

## Limitations

- 303 patients, 61 in test. The 25-split ranges are the real error bars; any difference under about 0.03 AUC is not measurable here.
- Coefficients are per standard deviation. For binary and one-hot columns that is not a natural unit, so those indicate direction and relative strength only.
- These are regularised coefficients, shrunk by the L2 penalty. Not maximum-likelihood estimates, no standard errors or confidence intervals. `statsmodels` is the right tool for inference.
- Collinearity among the exercise-test variables was not formally checked. `oldpeak`, `slope`, `exang` and `thalach` all measure the same stress test; regularisation stabilises the fit but redistributes coefficient mass among correlated predictors.
- Observational data, so every coefficient is an association. Reverse causation is live: `ca` comes from fluoroscopy, which is ordered because disease is already suspected.
- The Cleveland data dates from 1988. None of these numbers estimate present-day performance.
- Nothing here is a diagnostic tool. It is a classification exercise on a public teaching dataset.

## Stack

Python, pandas, NumPy, scikit-learn (Pipeline, SimpleImputer, StandardScaler, LogisticRegression, KNeighborsClassifier, RandomForestClassifier, GridSearchCV, RepeatedStratifiedKFold, StratifiedShuffleSplit), matplotlib, seaborn.

## Files

- `notebook.ipynb`: the full analysis with executed outputs and figures.