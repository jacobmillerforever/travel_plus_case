# ----------------------------------------------------------------------------
# Assignment 1 (IDS 572) - TravelPlus RoadsidePlus Case Study
# Contributor: Analyst 2 - Data Preparation & Predictive Modeling Lead
#
# Objective:
#   Define the full modeling workflow that transforms raw data into trained
#   predictive models, documents tuning rationale, and produces evaluation
#   outputs ready for downstream marketing analysis.
#
# Assumptions:
#   - Analyst 1's EDA findings are available to inform preprocessing choices.
#   - Data is stored in data/dataTravelPlus.csv (contacted customers only).
#   - Modeling will be executed in R using packages such as tidymodels, caret,
#     or equivalent; specific implementation left to analyst discretion.
#
# Key Deliverables for this Script:
#   1. Clearly documented preprocessing pipeline (handling of missing data,
#      transformations, encoding, feature selection).
#   2. Train/test (or train/validation/test) split strategy with rationale.
#   3. Trained and tuned models for logistic regression, k-NN, decision tree
#      (CART or C5.0), random forest, and gradient boosting (GBM/XGBoost/etc.).
#   4. Standard evaluation metrics for each model (confusion matrix statistics,
#      ROC/AUC) on hold-out data.
#   5. Variable importance analysis across models with interpretive commentary.
#
# ---------------------------------------------------------------------------
# 1) Load packages and data
#   - Load tidyverse for manipulation, tidymodels/caret for modeling, plus
#     recipes/workflows packages if using tidymodels framework.
#   - Import the dataset and perform initial checks (dimensions, types) to ensure
#     consistency with Analyst 1's EDA outputs.
#
# 2) Define target and feature sets
#   - Explicitly set Purchase as the binary outcome and convert to factor with
#     meaningful level ordering (e.g., "No" < "Yes").
#   - Confirm all predictors are correctly typed (numeric vs. categorical) and
#     align with modeling requirements identified during EDA.
#
# 3) Handle missing data and anomalies
#   - Document missing data handling strategy (e.g., median/mean imputation,
#     mode imputation, k-NN imputation, or dropping variables with excessive
#     missingness).
#   - Address outliers or skewness where necessary (transformations, winsorizing,
#     etc.) and note justifications in comments.
#
# 4) Feature engineering / selection
#   - Consider creating interaction terms or derived ratios flagged by EDA.
#   - Decide whether to drop redundant variables (e.g., composite scores deemed
#     unnecessary) or apply dimensionality reduction techniques.
#   - Standardize/normalize numeric predictors when required by specific models
#     (e.g., k-NN, logistic regression with regularization).
#   - Encode categorical variables (one-hot/dummy encoding or other methods) and
#     ensure resulting design matrix is compatible with all model types.
#
# 5) Partition the data
#   - Split data into training and testing sets (e.g., 70/30) with stratification
#     on Purchase to maintain class balance.
#   - Optionally create cross-validation folds within the training set for tuning.
#   - Document random seed usage for reproducibility.
#
# 6) Address class imbalance
#   - Evaluate if Purchase imbalance requires remediation (e.g., SMOTE, downsampling,
#     upweighting) and apply appropriate techniques within cross-validation to
#     avoid data leakage.
#   - Record rationale for chosen approach and any trade-offs considered.
#
# 7) Model training and tuning
#   - For each model type, outline the full workflow:
#       * Logistic Regression: specify whether using standard GLM, penalized (lasso,
#         ridge, elastic net); tune relevant hyperparameters; assess coefficients.
#       * k-NN: scale features, tune k using cross-validation, consider distance
#         metrics and weighting schemes.
#       * Decision Tree: decide between CART (rpart) or C5.0; tune complexity
#         parameters (cp, maxdepth, minsplit, boosting trials for C5.0).
#       * Random Forest: tune number of trees, mtry, minimum node size; capture
#         out-of-bag error estimates.
#       * Gradient Boosting Machine: choose gbm/xgboost/lightgbm implementation;
#         tune learning rate, tree depth, number of trees, subsampling rates.
#   - For each model, capture tuning grids, validation performance, and chosen
#     final hyperparameters in comments or exported tables.
#
# 8) Evaluate model performance
#   - Generate predictions on the hold-out test set for each finalized model.
#   - Compute confusion matrix metrics (accuracy, precision, recall, specificity,
#     F1-score) and compare across models.
#   - Plot ROC curves and calculate AUC; store figures/tables for reporting.
#   - Summarize key takeaways regarding which model(s) perform best overall.
#
# 9) Investigate variable importance
#   - Extract variable importance measures appropriate to each model:
#       * Logistic regression: standardized coefficients or odds ratios.
#       * k-NN: assess via model-agnostic methods (permutation importance, SHAP).
#       * Trees/forest/GBM: use built-in importance metrics and validate with
#         permutation-based approaches.
#   - Compare importance rankings across models; discuss consistent top predictors
#     versus method-specific signals.
#   - Document any surprising findings (e.g., lifestyle variables surfacing as
#     important despite initial skepticism).
#
# 10) Summarize modeling insights
#   - Compile a concise report-ready summary: best-performing model(s), key
#     predictors, treatment of imbalance, and limitations.
#   - Explicitly prepare outputs (model objects, metrics tables) needed by Analyst 3
#     for lift/incremental analysis and business recommendation.
# ----------------------------------------------------------------------------
