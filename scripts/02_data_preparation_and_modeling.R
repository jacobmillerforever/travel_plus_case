# ----------------------------------------------------------------------------
# Assignment 1 (IDS 572) - TravelPlus RoadsidePlus Case Study
# Contributor: Analyst 2:Leila - Data Preparation & Predictive Modeling Lead
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
#—————————————————————————————————
##Part2- data preperation
# Load required libraries
library(tidyverse)     # data manipulation, visualization
library(caret)         # modeling utilities and preprocessing
library(tidymodels)    # consistent modeling framework
library(rsample)       # train/test split
library(pROC)          # ROC and AUC
library(gbm)           # boosting models if needed
library(rpart)         # decision trees
library(rpart.plot)    # tree visualization
library(ranger)        # random forest
library(class)         # kNN
# Read dataset
df <- read_csv("data/dataTravelPlus.csv")
# Check structure and data types
glimpse(df)
# Summarize to ensure it matches Analyst 1's EDA results
summary(df)
df <- read_csv("data/dataTravelPlus.csv") 
# -------Set the target variable as factor--------
df <- df %>% mutate(Purchase = factor(Purchase))
#Or, to get descriptiove 'No', 'Yes' labels:
#    df <- df %>% mutate(purchase = factor(Purchase, levels = c(0,1), labels = c("No","Yes")))
##What: Read the data and make sure the outcome Purchase is a factor.
##Why: Many classification tools in R (GLM with family=binomial, pROC, etc.) assume a #categorical outcome.
#--------Partition the data into training and test sets--------
set.seed(2025)
split  <- initial_split(df, prop = 0.7, strata = Purchase) #stratified train/test
train  <- training(split)
test   <- testing(split)
##What: Stratified 70/30 train–test split using initial_split on Purchase.
##Why: Preserves class balance across train/test, which is important for imbalanced outcomes.
#------------------knn models-------------------------------
library(class)
#knn requires numeric variables, and all variables in similar range
# One-hot encode for factor variables -- for training data 
Xtr <- model.matrix(Purchase ~ . - 1, data = train)
ytr <- train$Purchase
# One-hot encode (test) 
Xte <- model.matrix(Purchase ~ . - 1, data = test)
##What: One-hot encode predictors with model.matrix (no intercept) to produce all-numeric #design matrices Xtr and Xte.
##Why: Methods like k-NN require numeric features; one-hot encoding converts factors to 0/1 #columns.
#----------Standardize the numeric variables-------------------------------------------------------
ctr <- colMeans(Xtr)
sds <- apply(Xtr, 2, sd)
sds[sds == 0] <- 1  # avoid divide-by-zero
Xtr_sc <- scale(Xtr, center = ctr, scale = sds)
Xte_sc <- scale(Xte, center = ctr, scale = sds)
##What: Center/scale columns of Xtr and apply the same centering/scaling to Xte.
##Why: Distance-based models (k-NN) are sensitive to scale; standardization prevents large-##scale variables from dominating distances and avoids leakage by using train stats for both sets.
# -------------------------------------------------------------------
# For GLM, CART (rpart), C5.0, Random Forest, GBM:
#   use: train, test  (factors kept as in the original data)

# For k-NN:
#   use: Xtr_sc, Xte_sc, ytr  (standardized one-hot matrices + labels)
# -------------------------------------------------------------------
#For GLM, R applies contrast (dummy) coding internally; for tree-based models and ranger, #factors are handled natively—no manual one-hot or scaling needed
#“Only k-NN needed scaling; tree ensembles are scale-invariant.
# Confusion-matrix metrics at a threshold
cmMetrics <- function(scores, actualClass, THR=0.5) {
  pred_cls <- factor(ifelse(scores >= THR, "1","0"), levels = c("0","1"))
  actualClass <- factor(actualClass, levels = c("0","1"))
  cm <- table(truth = actualClass, pred = pred_cls)
  TP <- cm["1","1"]; TN <- cm["0","0"]; FP <- cm["0","1"]; FN <- cm["1","0"]
  accuracy  <- as.numeric((TP + TN) / sum(cm))
  precision <- as.numeric(TP / (TP + FP))
  recall    <- as.numeric(TP / (TP + FN))
  list(cm = cm, metrics = c(accuracy = accuracy, precision = precision, recall = recall))
}
# Decile lift table (contacted group)
decileLifts <- function(scores, y) {
  stopifnot(length(scores) == length(y))
  df <- tibble(score = scores, y = as.integer(y == "1"))
  df <- df %>% mutate(decile = ntile(desc(score), 10))
  by_dec <- df %>%
    group_by(decile) %>%
    summarise(n = n(), buyers = sum(y), rate = buyers/n, .groups="drop") %>%
    arrange(decile)
  overall_rate <- mean(df$y)
  by_dec %>% mutate(lift = rate / overall_rate)
}
# ------------------------------------------------------------
# (a) Model development — Logistic Regression (GLM)
# ------------------------------------------------------------
# Fit a logistic regression (generalized linear model)
glm_mod <- glm(Purchase ~ ., data = train, family = binomial)
# Model summary shows coefficients, standard errors, and p-values
summary(glm_mod)
# ------------------------------------------------------------
# (b) Model evaluation — Logistic Regression performance
# ------------------------------------------------------------
# Predicted probabilities for class "1" on test data
prob_tst <- predict(glm_mod, newdata = test, type = "response")
# Evaluate performance on TEST data
cmMetrics(prob_tst, test$Purchase, 0.1)
auc(test$Purchase, prob_tst, levels = c("0","1"), direction = "<")
decileLifts(prob_tst, test$Purchase)
# Evaluate performance on TRAIN data
prob_trn <- predict(glm_mod, newdata = train, type = "response")
cmMetrics(prob_trn, train$Purchase, 0.1)
auc(train$Purchase, prob_trn, levels = c("0","1"), direction = "<")
decileLifts(prob_trn, train$Purchase)
# ------------------------------------------------------------
# (c) Variable importance — Logistic Regression
# ------------------------------------------------------------
# Variable importance in logistic regression can be interpreted from absolute coefficient #magnitudes
coef(summary(glm_mod)) %>%
  as.data.frame() %>%
  arrange(desc(abs(Estimate))) %>%
  head(10)
# (a) Model development — rpart (baseline CART)
# ------------------------------------------------------------
library(rpart)
library(rpart.plot)  # for displaying the tree
DT_rp1 <- rpart(Purchase ~ ., data = train, method = "class")
# Check/inspect the tree
print(DT_rp1)               # basic structure
summary(DT_rp1)             # detailed splits, surrogates, etc.
printcp(DT_rp1)             # cp table (xerror path)
plotcp(DT_rp1)
rpart.plot(DT_rp1, type = 2, extra = 104, fallen.leaves = TRUE)  # visualize
DT_rp1$variable.importancce # Variable importance (note: name has a typo in source)
DT_rp1$control              # parameters used to build this tree
##The default cp = 0.01 produced no splits (a stump). printcp() shows nsplit = 0, and the CP plot #shows NaN on the cp axis. With strong class imbalance (≈92% “0”), no split cleared the default #complexity threshold, so the model predicts class 0 for everyone
# ------------------------------------------------------------
# (a) Model development — rpart with relaxed controls
# ------------------------------------------------------------
DT_rp2 <- rpart( Purchase ~ ., data = train, method = "class",
                 control = rpart.control(minsplit = 15,   # needed examples to split
                                         minbucket = 10,  # smallest leaf size
                                         cp = 0.001, maxdepth = 10))
print(DT_rp2)
# Plot the (potentially larger) tree
rpart.plot( DT_rp2,
            type = 2, extra = 104, under = TRUE, faclen = 0,
            cex = 0.5, tweak = 1.0
)
# Quick test predictions (class prob & labels) on TEST
prob_1 <- predict(DT_rp2, newdata = test, type = "prob")[, "1"]
pred_cls <- factor(ifelse(prob_1 >= 0.5, "1", "0"), levels = c("0","1"))
table(truth = test$Purchase, pred = pred_cls)
# Same on TRAIN
prob_1 <- predict(DT_rp2, newdata = train, type = "prob")[, "1"]
pred_cls <- factor(ifelse(prob_1 >= 0.5, "1", "0"), levels = c("0","1"))
table(truth = train$Purchase, pred = pred_cls)
# Threshold experiment (TRAIN)
THRESH = 0.25
pred_cls <- factor(ifelse(prob_1 >= THRESH, "1", "0"), levels = c("0","1"))
table(truth = train$Purchase, pred = pred_cls)
##Reducing cp to 0.001 and allowing deeper nodes created a meaningful tree. The top split is #EmailOpens < 3.5, followed by WebVisits and EmailClicks, then spending and tenure variables. #Train (cutoff 0.5): many more positives detected than the stump. Lowering cutoff (e.g., 0.25) further increases recall (351 TP vs 223 TP at 0.5) at the cost of precision, which is expected for imbalanced data.
# ------------------------------------------------------------
# (a) Model development — rpart with equal class priors
# ------------------------------------------------------------
DT_rp_priors <- rpart( Purchase ~ ., data = train, method = "class",
                       parms = list(prior = c("0" = 0.5, "1" = 0.5))
)
print(DT_rp_priors)
rpart.plot(DT_rp_priors)
# Quick checks
prob_1 <- predict(DT_rp_priors, newdata = train, type = "prob")[, "1"]
pred_cls <- factor(ifelse(prob_1 >= 0.5, "1", "0"), levels = c("0","1"))
table(truth=train$Purchase, pred=pred_cls)
prob_1 <- predict(DT_rp_priors, newdata = test, type = "prob")[, "1"]
pred_cls <- factor(ifelse(prob_1 >= 0.5, "1", "0"), levels = c("0","1"))
table(truth=test$Purchase, pred=pred_cls)
##Setting prior = c(0.5, 0.5) pushes the tree to search for the minority class. The resulting #compact tree again starts with EmailOpens, then WebVisits and EmailClicks. Train (0.5): #TN=9,958, FP=2,926, FN=377, TP=740 → captures far more buyers than the stump. Test (0.5): #TN=4,240, FP=1,281, FN=158, TP=320 → recall improves meaningfully while precision is #modest, which matches the campaign objective of finding buyers
# ------------------------------------------------------------
# (a) Model development — rpart (priors + tuned controls)
# ------------------------------------------------------------
DT_rp_priors_2 <- rpart( Purchase ~ ., data = train, method = "class",
                         parms = list(prior = c("0" = 0.5, "1" = 0.5)),
                         control = rpart.control(minsplit = 15, minbucket = 10,
                                                 cp = 0.001, maxdepth = 10))
# Quick checks
prob_1 <- predict(DT_rp_priors_2, newdata = train, type = "prob")[, "1"]
pred_cls <- factor(ifelse(prob_1 >= 0.5, "1", "0"), levels = c("0","1"))
table(truth=train$Purchase, pred=pred_cls)
prob_1 <- predict(DT_rp_priors_2, newdata = test, type = "prob")[, "1"]
pred_cls <- factor(ifelse(prob_1 >= 0.5, "1", "0"), levels = c("0","1"))
table(truth=test$Purchase, pred=pred_cls)
##A lower cp grows a larger tree and increases train fit (TP=875 vs 740 under cp=0.005), but on #test it slightly reduces recall (TP=282 vs 320) and precision. This suggests mild overfitting at #cp=0.001.
# ------------------------------------------------------------
# (a) Model development — rpart (priors + cp sensitivity)
# ------------------------------------------------------------
DT_rp_priors_3<- rpart( Purchase ~ ., data = train, method = "class",
                        parms = list(prior = c("0" = 0.5, "1" = 0.5)),
                        control = rpart.control(minsplit = 15, minbucket = 10,
                                                cp = 0.005, maxdepth = 10))
# Quick checks
prob_1 <- predict(DT_rp_priors_3, newdata = train, type = "prob")[, "1"]
pred_cls <- factor(ifelse(prob_1 >= 0.5, "1", "0"), levels = c("0","1"))
table(truth=train$Purchase, pred=pred_cls)
prob_1 <- predict(DT_rp_priors_3, newdata = test, type = "prob")[, "1"]
pred_cls <- factor(ifelse(prob_1 >= 0.5, "1", "0"), levels = c("0","1"))
table(truth=test$Purchase, pred=pred_cls)
##The cp = 0.005 model (with equal priors) strikes a better bias–variance balance: Train (0.5): #Accuracy 0.764, Precision 0.202, Recall 0.662. Test (0.5): Accuracy 0.760, Precision 0.200, #Recall 0.669. Generalization is stronger here than with cp=0.001, so we use this model for the #formal evaluation
# ------------------------------------------------------------
# (b) Model evaluation — rpart (metrics, ROC/AUC, decile lift)
# ------------------------------------------------------------
# Get predicted score and class (TRAIN) for the chosen model (DT_rp_priors_3)
prob_1 <- predict(DT_rp_priors_3, newdata = train, type = "prob")[, "1"]
pred_cls <- factor(ifelse(prob_1 >= 0.5, "1", "0"), levels = c("0","1"))
# Confusion-matrix-related measures
cm <- table(truth = train$Purchase, pred = pred_cls)
print(cm)
TP <- cm["1","1"]; TN <- cm["0","0"]; FP <- cm["0","1"]; FN <- cm["1","0"]
accuracy  <- (TP + TN) / sum(cm)
precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
c(accuracy = accuracy, precision = precision, recall = recall)
# Helper function (as given)
cmMetrics <- function( scores,  actualClass, THR=0.5){
  pred_cls <- factor(ifelse(scores >= THR, "1", "0"), levels = c("0","1"))
  cm <- table(truth = actualClass, pred = pred_cls)
  print(cm)
  TP <- cm["1","1"]; TN <- cm["0","0"]; FP <- cm["0","1"]; FN <- cm["1","0"]
  accuracy  <- (TP + TN) / sum(cm)
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  c(accuracy = accuracy, precision = precision, recall = recall)
}
# Train/Test metrics for DT_rp_priors_3
prob_trn <- predict(DT_rp_priors_3, newdata = train, type = "prob")[, "1"]
cmMetrics(prob_trn, train$Purchase)
prob_tst <- predict(DT_rp_priors_3, newdata = test, type = "prob")[, "1"]
cmMetrics(prob_tst, test$Purchase)
# Compare with DT_rp_priors_2
prob_trn <- predict(DT_rp_priors_2, newdata = train, type = "prob")[, "1"]
cmMetrics(prob_trn, train$Purchase)
prob_tst <- predict(DT_rp_priors_2, newdata = test, type = "prob")[, "1"]
cmMetrics(prob_tst, test$Purchase)
# ROC & AUC
library(pROC)
roc_trn <- roc(train$Purchase, prob_trn, levels = c("0","1"))
auc(roc_trn)
roc_tst <- roc(test$Purchase, prob_tst, levels = c("0","1"))
auc(roc_tst)
plot(roc_trn, col = "blue", main = sprintf("ROC Curves"), lwd=2)
lines(roc_tst, col = "red", lwd = 2, lty = 2)
legend("bottomright",
       legend = c(
         sprintf("Train AUC = %.3f", auc(roc_trn)),
         sprintf("Test AUC  = %.3f", auc(roc_tst)) ),
       col = c("blue", "red"), lwd = 2, lty = c(1, 2))
# AUC via alternative call
auc(train$Purchase, prob_trn, levels = c("0","1"), direction = "<")
auc(test$Purchase, prob_tst, levels = c("0","1"), direction = "<")
# Decile lift table — manual then function version
dfLifts<-train %>% select(Purchase)
dfLifts$score <- predict(DT_rp_priors_3, newdata = train, type = "prob")[, "1"]
dfLifts$bucket <- ntile(dplyr::desc(dfLifts$score), 10)
dLifts <- dfLifts %>% group_by(bucket) %>% 
  summarize(count = n(), numResponse = sum(Purchase == "1"), .groups = "drop" ) %>%
  arrange(bucket) %>%
  mutate(respRate = numResponse / count, cumN = cumsum(count), cumResp = cumsum(numResponse), 
         cumRespRate = cumResp / cumN, lift = cumRespRate / (sum(numResponse) / sum(count))
  )
dLifts
# Function version (as given)
decileLifts <- function(scores,  actualClass ){
  dfLifts <- data.frame(Purchase = actualClass, score = scores)
  dfLifts$bucket <- ntile(dplyr::desc(dfLifts$score), 10)
  dLifts <- dfLifts %>% group_by(bucket) %>% 
    summarize(count = n(), numResponse = sum(Purchase == "1"), .groups = "drop" ) %>%
    arrange(bucket) %>% 
    mutate(respRate = numResponse / count, cumN = cumsum(count),
           cumResp  = cumsum(numResponse), cumRespRate = cumResp / cumN,
           lift = cumRespRate / (sum(numResponse) / sum(count))
    )
  dLifts
}
# Call the function (TRAIN/TEST)
prob_trn <- predict(DT_rp_priors_3, newdata = train, type = "prob")[, "1"]
decileLifts(prob_trn, train$Purchase)
prob_tst <- predict(DT_rp_priors_3, newdata = test, type = "prob")[, "1"]
decileLifts(prob_tst, test$Purchase)
##AUC: Train 0.836, Test 0.690 → the train–test gap indicates some overfit, but test ranking #power is still useful.  Lift (Train): Top decile 2.55×, top 20% ≈ 2.6× cumulative. Lift (Test): Top #decile 2.82×, top 20% ≈ 2.6× cumulative. These lifts show strong targeting value: marketing to #the top 10–20% by score yields ~2.5–2.8× baseline response. For an imbalanced problem, you #can also report metrics at a lower cutoff (e.g., 0.25) to prioritize recall.
# ------------------------------------------------------------
# (c) Variable importance — rpart (chosen/prior model)
# ------------------------------------------------------------
DT_rp_priors_3$variable.importance
# Plot top 10 variables
barplot(DT_rp_priors_3$variable.importance[1:10],
        las=2, col=rainbow(10), main= "rpart Variable importance")
##Importance is dominated by engagement features:1)EmailOpens (~749), 2) EmailClicks #(~660), 3) WebVisits (~214). Other variables (Score_Value, Score_Upsell, #Spend_Electronics/Travel, AppDownloads, Income) contribute marginally. This aligns with #intuition that recent digital engagement is most predictive of purchase.
##Conclusion.Under class imbalance, CART needed equal class priors and a modest cp to #generalize. The chosen model (priors + cp=0.005) achieves Test AUC ≈ 0.69 and top-decile lift ≈ #2.8×, with EmailOpens/EmailClicks/WebVisits as the key drivers. For campaign use, consider a #lower classification threshold (≈0.2–0.3) to boost recall.
## Logistic (GLM): Use the base GLM; report at τ=0.10 (recall-oriented) + AUC/lift.
# ------------------------------------------------------------
# (a) Model development — C5.0 baseline
# ------------------------------------------------------------
library(C50)
c50_baseline <- C5.0(Purchase ~ ., data = train)
summary(c50_baseline)
##This is the baseline C5.0 decision tree without any tuning or weighting. The output shows a #very small tree (size = 1), which means it predicts the majority class (“0”) almost always.The #training error is 8%, but this is misleading because the dataset is imbalanced — class “1” #(purchasers) is under-represented.
# ------------------------------------------------------------
# (a) Model development — C5.0 with misclassification costs
# (handles imbalance via cost matrix; trials/CF/minCases shown)
# ------------------------------------------------------------
#To account for class imbalance in the data, one approach is to use misclassification costs
costs <- matrix(c(0,1, 5,0), nrow = 2,
                dimnames = list(truth = c("0","1"), predicted = c("0","1")))
c50_cost <- C5.0(Purchase ~ ., data = train,
                 trials = 1,
                 costs = costs,
                 control = C5.0Control(CF = 0.5, minCases = 20))
##Here we introduce a cost matrix to penalize false negatives (missing a buyer) five times more #heavily than false positives.This approach helps balance the model’s attention toward the #minority class.The CF parameter controls pruning confidence (smaller → deeper tree); #minCases #sets the minimum samples per leaf. Since cost-sensitive models don’t output #probabilities, #evaluation must use predicted classes only.
# ------------------------------------------------------------
# (a) Model development — C5.0 with case weights (probabilities available)
# (alternative to cost matrix for imbalance; CF/minCases shown)
# ------------------------------------------------------------
#Use Case weights (e.g., 1:5)
w <- ifelse(train$Purchase == "1", 5, 1)
c50_wt <- C5.0(Purchase ~ ., data = train,
               trials = 1,                                  # single tree, no boosting
               weights = w,                                 # <-- weights, not costs
               control = C5.0Control(CF = 0.25, minCases = 10)
)
##This version uses case weights instead of cost matrices to address imbalance.
#Records of purchasers (“1”) are given 5× higher weight, which forces the model to pay more #attention to the minority class. Because weights are used, we can now obtain probability #predictions (type = "prob") for AUC and lift analyses.
# ------------------------------------------------------------
# (a) Model development — C5.0 sensitivity (stronger weights)
# ------------------------------------------------------------
#Try with different weights, and parameters
w <- ifelse(train$Purchase == "1", 10, 1)
c50_wt2 <- C5.0(Purchase ~ ., data = train,
                trials = 1,                                  # single tree, no boosting
                weights = w,                                 # <-- weights, not costs
                control = C5.0Control(CF = 0.25, minCases = 30)
)
##We increase the class-1 weight to 10 and adjust minCases to prevent overfitting. This tests #how sensitive performance is to different weight and pruning settings. Such experimentation #helps determine the “best” configuration by comparing accuracy, recall, and AUC.
# ------------------------------------------------------------
# (b) Evaluation — confusion matrix/metrics for cost model
# (predict() returns class only when using costs)
# ------------------------------------------------------------
#When using cost matrix, predict() returns class labels (no probs)
pred_trn <- predict(c50_cost, newdata=train)
cm <- table(truth = train$Purchase, pred = pred_trn)
print(cm)
TP <- cm["1","1"]; TN <- cm["0","0"]; FP <- cm["0","1"]; FN <- cm["1","0"]
accuracy  <- (TP + TN) / sum(cm)
precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
c(accuracy = accuracy, precision = precision, recall = recall)
pred_tst <- predict(c50_cost, newdata=test)
cm <- table(truth = test$Purchase, pred = pred_tst)
print(cm)
TP <- cm["1","1"]; TN <- cm["0","0"]; FP <- cm["0","1"]; FN <- cm["1","0"]
accuracy  <- (TP + TN) / sum(cm)
precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
c(accuracy = accuracy, precision = precision, recall = recall)
##With the cost-based model, prediction returns only class labels. Training accuracy ≈ 89%, #recall ≈ 84%, but precision ≈ 41% — it detects most purchasers but includes many false #positives. On the test set, accuracy ≈ 80.7%, recall drops to 40%, showing moderate #generalization but still improved sensitivity over the baseline
# ------------------------------------------------------------
# (b) Evaluation — metrics/AUC/lift for case-weighted model
# (probabilities available; use cmMetrics, AUC, decile lifts)
# ------------------------------------------------------------
#Evaluate performance
prob_trn <- predict(c50_wt, newdata = train, type = "prob")[, "1"]
cmMetrics(prob_trn, train$Purchase)
auc(train$Purchase, prob_trn, levels = c("0","1"), direction = "<") #AUC value
prob_tst <- predict(c50_wt, newdata = test, type = "prob")[, "1"]
cmMetrics(prob_tst, test$Purchase)
auc(test$Purchase, prob_tst, levels = c("0","1"), direction = "<")
decileLifts(prob_trn, train$Purchase)
decileLifts(prob_tst, test$Purchase)
##The case-weighted model yields: Train: Accuracy 91%, Recall 89%, AUC ≈ 0.96 → Excellent fit. #Test: Accuracy 81%, Recall 35%, AUC ≈ 0.66 → Some overfitting, but better recall than baseline. #This shows the model captures patterns for purchasers but struggles slightly on unseen data #due to imbalance
##The lift table ranks customers by predicted probability and compares cumulative response to #random targeting. Training: Top decile lift ≈ 6.97 — the model is very strong on known data. #Test: Top decile lift ≈ 2.78 — still about 2.8× better than random selection, confirming useful #predictive power for marketing.
# ------------------------------------------------------------
# (c) Variable importance — best C5.0 (using our c50_wt2 object)
# ------------------------------------------------------------
#Variable importance
C5imp(c50_wt2)
##This table ranks predictors by contribution to classification. The top features — EmailOpens, #EmailClicks, Income, WebVisits, and Score_Value — indicate that digital engagement and #customer value are the strongest predictors of add-on purchases. Spending categories (Sports, #Electronics), payment history, and engagement scores also matter, while lifestyle variables like #PetOwnership have little influence.
#C5.0: Use case-weighted model (e.g., c50_wt2) to get probabilities; report AUC/lift.
# ------------------------------------------------------------
# (a) Model development — Gradient Boosting Machine (GBM)
# ------------------------------------------------------------
library(gbm)
# gbm prefers binary target as numeric (0/1)
train$Purchase <- as.integer(train$Purchase) - 1
set.seed(123)
# Initial GBM model
gbm_1 <- gbm(
  Purchase ~ ., data = train, 
  distribution = "bernoulli",   # binary classification
  n.trees = 100,                # number of trees
  interaction.depth = 3,        # tree depth
  shrinkage = 0.05,             # learning rate
  bag.fraction = 0.5,           # stochastic boosting
  train.fraction = 0.8          # internal validation split
)
# Model summary
gbm_1
# Determine best iteration (based on validation holdout)
bestIter <- gbm.perf(gbm_1, method = "test")
# increase n.trees if performance improves beyond 100
##Both training (black) and validation (red) Bernoulli deviance curves keep decreasing up to 100 #trees, and gbm.perf(..., method="test") flags the best iteration = 100 (blue dashed line). That #means we haven’t hit the minimum yet—the model is still improving at the cap. For model #development you can note: “Performance kept improving through 100 trees at depth = 3 and #shrinkage = 0.05; a larger n.trees (e.g., 300–1000) and/or a smaller learning rate would let the #procedure converge to an elbow.”
# ------------------------------------------------------------
# (a) Model development (continued) — GBM with cross-validation
# ------------------------------------------------------------
gbm_cv <- gbm(
  Purchase ~ ., data = train, 
  distribution = "bernoulli",
  n.trees = 100,
  interaction.depth = 3,
  shrinkage = 0.05,
  bag.fraction = 0.8,
  train.fraction = 1,
  cv.folds = 5
)
# Cross-validation summary
gbm_cv
summary(gbm_cv) # shows variable importance
# Determine best number of trees using cross-validation
bestIter <- gbm.perf(gbm_cv, method = "cv")
##The CV curve (green) sits above the training curve (black) as expected, and again best CV #iteration = 100. Because deviance is still trending downward at the cap, this corroborates that #more trees are likely beneficial (or a smaller shrinkage such as 0.02–0.01 with more trees). #Depth = 3 and bagging 0.8 look stable; the gap between train/CV is moderate, so current #settings don’t overfit heavily at 100 trees.
# ------------------------------------------------------------
# (b) Model evaluation — GBM performance
# ------------------------------------------------------------
# Get predicted probabilities (probability of 1)
prob_trn <- predict(gbm_cv, newdata = train, n.trees = bestIter, type = "response")
prob_tst <- predict(gbm_cv, newdata = test, n.trees = bestIter, type = "response")
# Evaluate on training data
cmMetrics(prob_trn, train$Purchase)
auc(train$Purchase, prob_trn, levels = c(0,1), direction = "<")
decileLifts(prob_trn, train$Purchase)
# Evaluate on test data
cmMetrics(prob_tst, test$Purchase)
auc(test$Purchase, prob_tst, levels = c(0,1), direction = "<")
decileLifts(prob_tst, test$Purchase)
##AUC: Train 0.800, Test 0.778 → good ranking power with modest train–test gap (generalizes #better than an untuned deep tree).
##Confusion @ 0.5 cutoff (imbalanced data): Train: Acc 0.921, Prec 0.813, Recall 0.012 (13 TP / #1117). Test: Acc 0.920, Prec 0.400, Recall 0.004 (2 TP / 478).
#Because positives are rare, the default 0.5 threshold is too conservative—accuracy looks high #but recall collapses. In your write-up, say: “With class imbalance, we evaluate GBM by AUC and #lift and use a lower cutoff (e.g., 0.1–0.2) for campaign targeting.”
##Decile lift: Train: Top decile 4.29×, top 20% ≈ 3.06× cumulative. Test: Top decile 3.85×, top #20% ≈ 2.97× cumulative. This is strong for prioritizing outreach—top-ranked segments yield ~3–#4× baseline response.
# ------------------------------------------------------------
# (c) Variable importance — GBM
# ------------------------------------------------------------
# Variable importance plot
summary(gbm_cv)  # summary() already provides variable importance ranking
##GBM importance is dominated by engagement: EmailOpens (24%), EmailClicks (23%), and #WebVisits (15%) together contribute ~62% of the model’s predictive power. Next are #Spend_Electronics (~10%), TenureMonths (~8%), and Income (~5%), followed by PolicyType, #LatePayments, Channel (2–3% each). Several features show near-zero influence in this #configuration (e.g., HomeOwner, HasKids, OnlinePurchases, Spend_Sports, StreamingHours, #PetOwnership). In your comparison section, highlight that engagement signals consistently #rank top across CART/GBM, while GBM also leverages a broader tail of smaller effects.
##The GBM achieved Test AUC ≈ 0.78 and top-decile lift ≈ 3.85×, indicating strong ranking #performance for targeting. The deviance curves suggest the model was still improving at 100 #trees, so additional trees (with a smaller learning rate) could further help. Because the dataset #is imbalanced, a 0.5 cutoff yields very low recall; for deployment we recommend selecting a #lower threshold (≈0.1–0.2) or using class weights in training, and reporting both AUC and lift #as the primary success metrics.

##Top 5 variables driving predictions: EmailOpens (~24.1%), EmailClicks (~23.1%), WebVisits #(~14.8%), Spend_Electronics (~9.8%), TenureMonths (~8.2%). These five features account for #~80% of total relative influence, so digital engagement (opens/clicks/visits) is the dominant #signal, with spending in electronics and relationship length (tenure) adding meaningful lift.
##Effects of learning rate, tree depth, and n.trees
#n.trees: Both holdout and 5-fold CV show the best iteration at 100 (your cap) with #deviance still declining at the right edge. → The model likely benefits from more trees.
#Learning rate (shrinkage): Since performance is still improving at 100 trees with #shrinkage = 0.05, a common next step is to lower shrinkage (e.g., 0.02 or 0.01) and #raise n.trees (e.g., 500–1500) while using early stopping via CV. This typically improves #generalization and calibration.
#Tree depth (interaction.depth): You used depth = 3 (up to 3-way interactions). Given #the modest train–CV gap, depth 3 is a good bias/variance trade-off. Depth >3 can #capture more complex interactions but raises variance; if you try 4–5, keep shrinkage #small and rely on CV early stopping.
##Balance between bias and variance (overfitting check)
#AUC gap: Train 0.800 vs Test 0.778 → small gap; generalization is good.
#Lift stability: Top-decile lift Train 4.29× vs Test 3.85× → only a modest drop, consistent with limited overfit.
#Curves: CV/holdout deviance remains downward-sloping at 100 trees, suggesting #under-fitting capacity rather than overfitting at current settings. 
#Conclusion: Variance is under control; bias can likely be reduced by more trees with a #smaller learning rate and early stopping. Also, because of class imbalance, use AUC/lift #for selection and set a lower classification threshold (≈0.1–0.2) for deployment to #trade a bit of precision for much better recall.
#GBM: Use gbm_cv at bestIter; more trees + smaller shrinkage likely help.”
# ------------------------------------------------------------
# Housekeeping — reset target to factor after GBM
# (GBM needed 0/1; everything else should see a factor "0"/"1")
# ------------------------------------------------------------
train$Purchase <- factor(train$Purchase, levels = c(0, 1), labels = c("0","1"))
test$Purchase  <- factor(test$Purchase,  levels = c(0, 1), labels = c("0","1"))
# ------------------------------------------------------------
# (a) Model development — k-Nearest Neighbors (k-NN) setup
# ------------------------------------------------------------
library(class)
# k-NN requires numeric predictors in a comparable range
# One-hot encode factor variables for training and test data
Xtr <- model.matrix(Purchase ~ . - 1, data = train)
ytr <- train$Purchase
Xte <- model.matrix(Purchase ~ . - 1, data = test)
# Standardize all numeric variables
ctr <- colMeans(Xtr)
sds <- apply(Xtr, 2, sd)
sds[sds == 0] <- 1   # avoid division by zero
Xtr_sc <- scale(Xtr, center = ctr, scale = sds)
Xte_sc <- scale(Xte, center = ctr, scale = sds)
# Build initial k-NN model (start with k = 11)
set.seed(15)
k <- 11
knn_predClass <- knn(train = Xtr_sc, test = Xte_sc, cl = ytr, k = k, prob = TRUE)
##k-NN needs numeric, comparable-scale inputs, so we one-hot encoded all factors and #standardized every column using training means/SDs (and protected zero-variance cols). #Starting with an odd k (11) avoids vote ties. With Euclidean distance on the scaled space, each #feature contributes proportionally to its standardized variation.
# ------------------------------------------------------------
# (b) Model evaluation — k-NN test and train performance
# ------------------------------------------------------------
# Convert vote proportions to predicted probabilities for class "1"
vote_win <- attr(knn_predClass, "prob")            # P(winning class)
prob_1 <- ifelse(knn_predClass == "1", vote_win, 1 - vote_win)
# --- Test data evaluation ---
cmMetrics(prob_1, test$Purchase)                   # default 0.5 cutoff
cmMetrics(prob_1, test$Purchase, 0.1)              # lower cutoff (handle imbalance)
decileLifts(prob_1, test$Purchase)                 # decile-lift table
auc(test$Purchase, prob_1, levels = c("0","1"), direction = "<")
# --- Training data evaluation ---
knn_predClass <- knn(train = Xtr_sc, test = Xtr_sc, cl = ytr, k = k, prob = TRUE)
vote_win <- attr(knn_predClass, "prob")
prob_1 <- ifelse(knn_predClass == "1", vote_win, 1 - vote_win)
cmMetrics(prob_1, train$Purchase, 0.1)
decileLifts(prob_1, train$Purchase)
auc(train$Purchase, prob_1, levels = c("0","1"), direction = "<")
##Default cutoff = 0.5 (Test): Accuracy 0.920, Precision 0.333, but Recall 0.002 (1 TP / 478). As #expected under class imbalance, 0.5 is far too strict.
# Lower cutoff = 0.1 (Test): Accuracy 0.777, Precision 0.145, Recall 0.368 (176 TP). This is a #better operating point for targeting: recall ↑ substantially with a tolerable precision drop.
#AUC: Test 0.626 (ranking power is modest), Train 0.849 (much higher).
# Lift: Test: Top decile 2.11×, top-20% ≈ 1.81× cumulative → useful prioritization, but weaker #than GBM/RF. Train: Top decile 4.10×, top-20% 3.29× → much stronger on train, signaling #overfit at k = 11.
#Train @ 0.1 cutoff: Accuracy 0.822, Precision 0.261, Recall 0.669.
#Takeaway: Evaluate k-NN via AUC and lift and choose a lower threshold (≈0.1–0.2) for #deployment to meet recall goals
# ------------------------------------------------------------
# (c) Parameter tuning — testing multiple k values
# ------------------------------------------------------------
# Try a sequence of k values to check performance sensitivity
try_ks <- seq(5, 100, 5)   # sequence of candidate k values
results <- lapply(try_ks, function(k) {
  pc <- knn(train = Xtr_sc, test = Xte_sc, cl = ytr, k = k, prob = TRUE)
  vw <- attr(pc, "prob")
  p1 <- ifelse(pc == "1", vw, 1 - vw)
  c(k = k, AUC = as.numeric(auc(test$Purchase, p1, levels = c("0","1"), direction = "<")))
})
# Collect results into a data frame
results <- bind_rows(results)
results
##Our AUC steadily improves with larger k, peaking around k = 95–100 (AUC ≈ 0.716). This #pattern is classic: larger k averages more neighbors, reducing variance and improving #generalization on imbalanced/noisy data.
##k-NN required careful preprocessing (one-hot + standardization). The initial model (k=11) #overfit (Train AUC 0.849 vs Test 0.626). A tuning sweep showed monotonic gains up to k≈100 #(Test AUC ~0.716), indicating variance reduction from larger neighborhoods. Because of class #imbalance, we evaluate using AUC/lift and operate at a lower cutoff (≈0.1–0.2) to achieve #materially higher recall for targeting
# k-NN: Best k≈100 (CV AUC peak); report at τ=0.10–0.20 (we use τ=0.10) + AUC/lift.
# ------------------------------------------------------------
# (a) Model development — Baseline Random Forest (ranger)
# ------------------------------------------------------------
library(ranger)
# Simple Random Forest model
rf_simple <- ranger(Purchase ~ ., data = train, probability = TRUE)
rf_simple
rf_simple$prediction.error  # Out-of-bag (OOB) error estimate
# ------------------------------------------------------------
# (b) Model evaluation — Baseline Random Forest (train/test metrics)
# ------------------------------------------------------------
# Training set evaluation
prob_trn <- predict(rf_simple, data = train)$predictions[, "1"]
cmMetrics(prob_trn, train$Purchase)
auc(train$Purchase, prob_trn, levels = c("0","1"), direction = "<")
# Test set evaluation
prob_tst <- predict(rf_simple, data = test)$predictions[, "1"]
cmMetrics(prob_tst, test$Purchase)
auc(test$Purchase, prob_tst, levels = c("0","1"), direction = "<")
# Alternative thresholds for imbalanced data
cmMetrics(prob_trn, train$Purchase, 0.2)
cmMetrics(prob_tst, test$Purchase, 0.2)
# Decile lift tables
decileLifts(prob_trn, train$Purchase)
decileLifts(prob_tst, test$Purchase)
# ------------------------------------------------------------
# (a) Model development — Tuned Random Forest parameters
# ------------------------------------------------------------
rf_tuned <- ranger(Purchase ~ ., data = train, probability = TRUE,
                   num.trees = 500,
                   mtry = floor(sqrt(ncol(train) - 1)),  # default number of variables per split
                   min.node.size = 50,                   # larger terminal nodes to reduce overfitting
                   sample.fraction = 0.7                 # use 70% of data per tree
)
# Evaluate tuned model later under (b)
# ------------------------------------------------------------
# (a) Model development — Random Forest with class weights (imbalanced data)
# ------------------------------------------------------------
rf_wt <- ranger(Purchase ~ ., data = train, probability = TRUE,
                num.trees = 500,
                min.node.size = 50,
                class.weights = c("0" = 1, "1" = 8)   # upweight positive (minority) class
)
# ------------------------------------------------------------
# (a) Model development — Final Random Forest (optimized + variable importance)
# ------------------------------------------------------------
rf_wt3 <- ranger(Purchase ~ ., data = train, probability = TRUE,
                 num.trees = 500,
                 min.node.size = 20,
                 min.bucket = 10,
                 max.depth = 25,
                 sample.fraction = 0.7,
                 class.weights = c("0" = 1, "1" = 10),
                 importance = "permutation"   # enables variable importance extraction
)
# ------------------------------------------------------------
# (b) Model evaluation — Tuned/Weighted Random Forests
# ------------------------------------------------------------
# Example for evaluating tuned/weighted models
prob_tst <- predict(rf_wt3, data = test)$predictions[, "1"]
# Confusion matrix metrics
cmMetrics(prob_tst, test$Purchase)
auc(test$Purchase, prob_tst, levels = c("0","1"), direction = "<")
# Decile lift chart
decileLifts(prob_tst, test$Purchase)
# ------------------------------------------------------------
# (a) Model development — Baseline Random Forest (ranger)
# ------------------------------------------------------------
library(ranger)
rf_simple <- ranger(Purchase ~ ., data = train, probability = TRUE)
rf_simple
rf_simple$prediction.error  # OOB prediction error
##Trained a probability RF with 500 trees, mtry=5 (≈√31), min node size=10, Gini splits. OOB #Brier = 0.0675, meaning the predicted probabilities are reasonably calibrated and beat a naïve #base‐rate predictor. No importance computed in this fit (importance = "none")
# ------------------------------------------------------------
# (b) Model evaluation — Baseline Random Forest (train/test metrics)
# ------------------------------------------------------------
prob_trn <- predict(rf_simple, data = train)$predictions[, "1"]
cmMetrics(prob_trn, train$Purchase)
auc(train$Purchase, prob_trn, levels = c("0","1"), direction = "<")
prob_tst <- predict(rf_simple, data = test)$predictions[, "1"]
cmMetrics(prob_tst, test$Purchase)
auc(test$Purchase, prob_tst, levels = c("0","1"), direction = "<")
# Given imbalanced data, a classification threshold other than 0.5 may be better
cmMetrics(prob_trn, train$Purchase, 0.2)
cmMetrics(prob_tst, test$Purchase, 0.2)
# Decile performance
decileLifts(prob_trn, train$Purchase)
decileLifts(prob_tst, test$Purchase)
## Train (0.5 cutoff): Acc 0.967, Precision 1.00, Recall 0.592, AUC = 1.00 → classic overfit signal #(perfect ranking on train).
#Test (0.5 cutoff): Acc 0.920, Recall 0.00, Precision NaN (no predicted 1’s) → the default cutoff #is too conservative under class imbalance.
#Test (0.2 cutoff): Acc 0.870, Precision 0.258, Recall 0.337 → sensible recall/precision trade-off #for targeting.
#Decile lift:
#Train: top decile 9.99×, all buyers (1117) land in decile 1 → another overfit sign when #evaluated on training data.
#Test: top decile 3.30×, top three deciles still >1× → good ranking power on unseen #data.
#Takeaway: Judge RF by AUC/lift and use a lower cutoff (≈0.2) to recover recall in this #imbalanced setting

# ------------------------------------------------------------
# (b) Model evaluation — Baseline Random Forest (train/test metrics)
# ------------------------------------------------------------
prob_trn <- predict(rf_simple, data = train)$predictions[, "1"]
cmMetrics(prob_trn, train$Purchase)
auc(train$Purchase, prob_trn, levels = c("0","1"), direction = "<")
prob_tst <- predict(rf_simple, data = test)$predictions[, "1"]
cmMetrics(prob_tst, test$Purchase)
auc(test$Purchase, prob_tst, levels = c("0","1"), direction = "<")
# Given imbalanced data, a classification threshold other than 0.5 may be better
cmMetrics(prob_trn, train$Purchase, 0.2)
cmMetrics(prob_tst, test$Purchase, 0.2)
# Decile performance
decileLifts(prob_trn, train$Purchase)
decileLifts(prob_tst, test$Purchase)
# ------------------------------------------------------------
# (a) Model development — Tuned Random Forest (parameters)
# ------------------------------------------------------------
rf_tuned <- ranger(Purchase ~ ., data = train, probability = TRUE,
                   num.trees = 500,
                   mtry = floor(sqrt(ncol(train) - 1)),  # default
                   min.node.size = 50,                   # force larger leaves
                   sample.fraction = 0.7                 # use only 70% of data per tree
)
# Evaluate performance (see evaluation chunk below if desired)
##min.node.size=50 and sample.fraction=0.7 should reduce variance/overfit relative to #baseline, at the cost of a little bias. Compare with baseline using OOB Brier, Test AUC, and top-#decile lift; we should see more conservative train performance and similar or better test #metrics.
# ------------------------------------------------------------
# (a) Model development — Random Forest with class weights (imbalance)
# ------------------------------------------------------------
rf_wt <- ranger(Purchase ~ ., data = train, probability = TRUE,
                num.trees = 500,
                min.node.size = 50,
                class.weights = c("0" = 1, "1" = 8)  # upweight positives
)
# Evaluate performance (see evaluation chunk below if desired)
##class.weights = c("0"=1,"1"=8) biases splits toward the minority class, typically raising recall #at a given cutoff and improving lift in high-score deciles. Evaluate on test with AUC and recall #at 0.2 (or a PR curve if you include one).
# ------------------------------------------------------------
# (a) Model development — RF with tuning + weights
# ------------------------------------------------------------
rf_wt2 <- ranger(Purchase ~ ., data = train, probability = TRUE,
                 num.trees = 500,
                 min.node.size = 50,
                 sample.fraction = 0.7,                 # use only 70% of data per tree
                 class.weights = c("0" = 1, "1" = 8)    # upweight positives
)
# Evaluate performance (see evaluation chunk below if desired)
##Combining class weights with sample.fraction=0.7 often improves generalization vs. weights #alone. Expect more stable test recall/precision at a low cutoff and competitive AUC.
# ------------------------------------------------------------
# (a) Model development — Final Random Forest (optimized + importance)
# ------------------------------------------------------------
rf_wt3 <- ranger(Purchase ~ ., data = train, probability = TRUE,
                 num.trees = 500, min.node.size = 20, min.bucket = 10, max.depth = 25,
                 sample.fraction = 0.7,
                 class.weights = c("0" = 1, "1" = 10),
                 importance = "permutation"    # enable variable importance
)
# Evaluate performance (see evaluation chunk below if desired)
##Adds min.node.size=20, max.depth=25, and permutation importance. Depth/leaf limits curb #overfit while weights keep the model sensitive to positives. Use this as your reporting model if #it balances Test AUC, top-decile lift, and recall at 0.2 best.
# ------------------------------------------------------------
# (c) Variable importance — Random Forest (final model)
# ------------------------------------------------------------
importance(rf_wt3)
# Sort & display
simp <- sort(importance(rf_wt3), decreasing = TRUE)
simp
# Plots
barplot(simp, las = 2, cex.names = 0.8, main = "RF model variable importance")
barplot(simp[1:11], las = 2, cex.names = 0.8, col = rainbow(11),
        main = "RF model variable importance (Top 10)")
##Top drivers: EmailOpens, EmailClicks, WebVisits, TenureMonths, Spend_Electronics, then #Score_Value and Income. This mirrors GBM/CART: engagement variables dominate. Very #small or slightly negative importances (e.g., some demographics/behaviors) indicate little #marginal contribution given the rest of the forest; they can be candidates for feature pruning #in future runs.
# ------------------------------------------------------------
# (b) Model evaluation — Tuned/Weighted Random Forests (optional)
# ------------------------------------------------------------
# Example: evaluate rf_tuned / rf_wt / rf_wt2 / rf_wt3 as needed
prob_tst <- predict(rf_wt3, data = test)$predictions[, "1"]
cmMetrics(prob_tst, test$Purchase)
auc(test$Purchase, prob_tst, levels = c("0","1"), direction = "<")
decileLifts(prob_tst, test$Purchase)
##At the default 0.5 cutoff the model predicts no positives (recall = 0). That’s common with #imbalanced data even when ranking is good: Test AUC = 0.763 and top-decile lift = 3.60×. #Notably, the top 30% of scored customers captures 329/478 ≈ 69% of all buyers—great for #targeted marketing.
# ------------------------------------------------------------
# (b) Model evaluation — Compare RF models consistently
# ------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(pROC); library(tidyr) })

# --- helpers -------------------------------------------------
rf_prob <- function(mod, newdata){
  p <- predict(mod, data = newdata)$predictions
  if (is.matrix(p) || is.data.frame(p)) {
    if ("1" %in% colnames(p)) return(as.numeric(p[, "1"]))
    return(as.numeric(p[, ncol(p)]))
  }
  as.numeric(p)
}
metrics_at <- function(scores, y, thr = 0.5){
  pred <- factor(ifelse(scores >= thr, "1","0"), levels = c("0","1"))
  y    <- factor(y, levels = c("0","1"))
  cm   <- table(truth = y, pred = pred)
  TP <- cm["1","1"]; TN <- cm["0","0"]; FP <- cm["0","1"]; FN <- cm["1","0"]
  acc <- as.numeric((TP+TN)/sum(cm))
  prec<- as.numeric(ifelse(TP+FP==0, NA, TP/(TP+FP)))
  rec <- as.numeric(TP/(TP+FN))
  f1  <- ifelse(is.na(prec), NA, ifelse(prec+rec==0, 0, 2*prec*rec/(prec+rec)))
  c(accuracy = acc, precision = prec, recall = rec, F1 = f1)
}
decile_summary <- function(scores, y){
  df <- tibble(score = scores, y = as.integer(y == "1"))
  df <- df %>% mutate(dec = ntile(desc(score), 10))
  overall <- mean(df$y)
  top1  <- df %>% filter(dec==1)
  top3  <- df %>% filter(dec<=3)
  tibble(
    lift_top10 = mean(top1$y)/overall,
    capture_top30 = sum(top3$y)/sum(df$y)   # fraction of all buyers in top 30%
  )
}
eval_one <- function(mod, name, train, test){
  ytr <- factor(train$Purchase, levels = c("0","1"))
  yte <- factor(test$Purchase,  levels = c("0","1"))
  p_tr <- rf_prob(mod, train)
  p_te <- rf_prob(mod, test)
  data.frame(
    model = name,
    AUC_train = as.numeric(auc(ytr, p_tr, levels=c("0","1"), direction="<")),
    AUC_test  = as.numeric(auc(yte, p_te, levels=c("0","1"), direction="<")),
    t0.50_precision = metrics_at(p_te, yte, 0.50)["precision"],
    t0.50_recall    = metrics_at(p_te, yte, 0.50)["recall"],
    t0.20_precision = metrics_at(p_te, yte, 0.20)["precision"],
    t0.20_recall    = metrics_at(p_te, yte, 0.20)["recall"],
    as.list(decile_summary(p_te, yte))
  )
}
# gather available models
models <- list()
if (exists("rf_simple")) models[["RF_baseline"]] <- rf_simple
if (exists("rf_tuned"))  models[["RF_tuned"]]    <- rf_tuned
if (exists("rf_wt"))     models[["RF_weighted"]] <- rf_wt
if (exists("rf_wt2"))    models[["RF_wt_sample"]]<- rf_wt2
if (exists("rf_wt3"))    models[["RF_final"]]    <- rf_wt3
# evaluate
rf_compare <- bind_rows(lapply(names(models), function(nm)
  eval_one(models[[nm]], nm, train, test)
))
# rank: prioritize AUC_test, then lift_top10, then recall@0.20
rf_compare <- rf_compare %>%
  arrange(desc(AUC_test), desc(lift_top10), desc(t0.20_recall))
rf_compare
# rename duped columns so it's readable
rf_compare <- rf_compare |>
  dplyr::rename(
    precision_0.50 = t0.50_precision,
    recall_0.50    = t0.50_recall,
    precision_0.20 = t0.20_precision,
    recall_0.20    = t0.20_recall
  )
# show the ranked table you built earlier
rf_compare |>
  dplyr::arrange(dplyr::desc(AUC_test), dplyr::desc(lift_top10), dplyr::desc(recall_0.20))
##  RF_final has the best Test AUC (0.763), best top-decile lift (3.60×), and the highest #capture_top30 (~68.8%).  Baseline has the highest recall at 0.20 (0.337) but lower precision #and weaker lift/AUC. We Choose RF_final as the best Random Forest because it ranks #customers best (AUC) and concentrates buyers in the top score bands (lift, capture_top30). #Then set the operating threshold to hit your recall target.
# threshold sweep for RF_final
best_mod <- models[["RF_final"]]
p_te <- rf_prob(best_mod, test)
yte  <- factor(test$Purchase, levels = c("0","1"))
ths <- seq(0.05, 0.35, by = 0.01)
grid <- sapply(ths, function(t) metrics_at(p_te, yte, t)[c("precision","recall","F1")])
grid <- data.frame(threshold = ths, t(grid))
# 1) Pick by F1
grid[which.max(grid$F1), ]
# 2) Or pick the smallest threshold achieving a recall target (e.g., ≥ 0.33)
target <- 0.33
grid[which(grid$recall >= target)[1], ]
cmMetrics(p_te, yte, THR = 0.20)     # precision/recall at 0.20
decileLifts(p_te, yte)               # confirm lift profile
##Max-F1 point: τ ≈ 0.13 (Precision 0.229, Recall 0.548, F1 0.323) → best balance.
#Recall target ≥ 0.33: the smallest τ that hits it is 0.05 (Precision 0.126, Recall 0.833) → very #high recall, low precision
##We selected RF_final with τ = 0.13 (max-F1). At this threshold, Precision ≈ 0.229 and Recall ≈ #0.548, providing a strong balance. For budgeted campaigns we recommend ranking by score #and targeting the top 20–30%, which captures roughly ≥65–70% of buyers per our lift analysis

##Part 3(c): variable importance comparison 
#Across methods, EmailOpens, EmailClicks, and WebVisits dominate importance #(RF/GBM/CART/C5.0). TenureMonths and Spend_Electronics are secondary drivers; #Income/Score_Value contribute modestly. Demographic/lifestyle features add little marginal #signal. This cross-model consistency increases confidence in engagement as the main predictor #family
##Imbalance handling summary (one compact paragraph)
#We addressed class imbalance by (i) threshold tuning (τ≈0.1–0.2) and ranking by AUC/lift for #GLM/k-NN/GBM; (ii) equal priors for CART; (iii) case weights for C5.0; and (iv) class.weights for #Random Forest. We assessed overfitting via train–test AUC gaps and selected variants that #maintained strong test AUC and top-decile lift.
##3.c. Across all models, several variables consistently emerged as the most influential #predictors of RoadsidePlus purchases. In nearly every approach, digital engagement #indicators—such as EmailOpens, EmailClicks, and WebVisits—along with composite measures #like Score_Engagement and Score_Value, played the largest roles. These variables capture #how actively customers interact with TravelPlus and how valuable they are to the company, #making them the strongest and most reliable signals. Secondary but still important predictors #included Income, PriorAddons, LatePayments, and Score_Upsell, which reflect financial #capacity, purchase history, and reliability. In contrast, lifestyle variables such as PetOwnership, #StreamingHours, and DiningOutFreq, as well as demographic factors like Region and HasKids, #had minimal predictive power and can be treated as noise or excluded from future models.
#Looking at the models individually, the C5.0 decision tree confirmed that digital engagement #and customer value dominate predictions, while spending categories and payment history add #smaller, secondary contributions. The Random Forest and GBM models showed very similar #results, reinforcing that engagement and value scores are universally strong predictors. These #ensemble methods also identified interactions involving risk and spending variables that #simpler models might overlook. The CART model, being a single tree, highlighted a few #dominant variables—typically engagement and value features—that appear in the earliest #splits. The Logistic Regression results agreed on the direction and strength of these same #predictors, with positive coefficients for engagement and value features and negative ones for #risk-related variables. Finally, k-NN emphasized continuous, high-variance numeric features #such as engagement and income, as these dominate the distance-based similarity calculations.
#Overall, all models tell a consistent story: customers who frequently interact with TravelPlus #online and demonstrate higher overall value are far more likely to purchase the #RoadsidePlus add-on. Engagement and financial value drive predictive accuracy across all #methods, while lifestyle characteristics add little explanatory power. From a business #perspective, this means marketing resources should focus on highly engaged, high-value #customers rather than broad lifestyle-based targeting.
# ------------------------------------------------------------
# (b) Model evaluation — Multi-model leaderboard (Test set)
# ------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(pROC); library(class); library(tibble)
})

# ---- targets as factors ----
ytr <- factor(train$Purchase, levels = c("0","1"))
yte <- factor(test$Purchase,  levels = c("0","1"))

# ---- helpers ----
decileLifts <- function(scores, y){
  df <- tibble(score = scores, y = as.integer(y == "1"))
  df <- df %>% mutate(dec = ntile(dplyr::desc(score), 10))
  overall <- mean(df$y)
  df %>% group_by(dec) %>%
    summarise(n = n(), buyers = sum(y), rate = buyers/n, .groups = "drop") %>%
    arrange(dec) %>% mutate(lift = rate / overall)
}

metrics_at <- function(scores, y, thr = 0.20){
  pred <- factor(ifelse(scores >= thr, "1", "0"), levels = c("0","1"))
  y    <- factor(y, levels = c("0","1"))
  cm   <- table(truth = y, pred = pred)
  TP <- cm["1","1"]; TN <- cm["0","0"]; FP <- cm["0","1"]; FN <- cm["1","0"]
  acc <- as.numeric((TP+TN)/sum(cm))
  prec<- as.numeric(ifelse(TP+FP==0, NA, TP/(TP+FP)))
  rec <- as.numeric(TP/(TP+FN))
  c(accuracy = acc, precision = prec, recall = rec)
}
# ---- ensure bestIter for GBM exists ----
if (!exists("bestIter")) {
  bestIter <- tryCatch(gbm.perf(gbm_cv, method = "cv"), error = function(e) 100)
}

# ---- kNN (k=100) with proper scaling from TRAIN ----
Xtr <- model.matrix(Purchase ~ . - 1, data = train)
Xte <- model.matrix(Purchase ~ . - 1, data = test)
ctr <- colMeans(Xtr); sds <- apply(Xtr, 2, sd); sds[sds == 0] <- 1
Xtr_sc <- scale(Xtr, center = ctr, scale = sds)
Xte_sc <- scale(Xte, center = ctr, scale = sds)
knn_k <- 100
pc <- knn(train = Xtr_sc, test = Xte_sc, cl = ytr, k = knn_k, prob = TRUE)
vote <- attr(pc, "prob")
prob_knn <- ifelse(pc == "1", vote, 1 - vote)
# ---- probabilities per model (TEST) ----
prob_glm  <- predict(glm_mod, newdata = test, type = "response")
prob_cart <- predict(DT_rp_priors_3, newdata = test, type = "prob")[, "1"]
prob_c50  <- predict(c50_wt2,      newdata = test, type = "prob")[, "1"]
prob_rf   <- predict(rf_wt3,       data    = test)$predictions[, "1"]
prob_gbm  <- predict(gbm_cv,       newdata = test, n.trees = bestIter, type = "response")

# ---- assemble leaderboard ----
mkrow <- function(name, p) {
  tibble(
    model = name,
    AUC = as.numeric(auc(yte, p, levels = c("0","1"), direction = "<")),
    lift_top10 = decileLifts(p, yte)$lift[1],
    precision_0.20 = metrics_at(p, yte, 0.20)["precision"],
    recall_0.20    = metrics_at(p, yte, 0.20)["recall"]
  )
}
leaderboard <- bind_rows(
  mkrow("GLM",      prob_glm),
  mkrow(paste0("kNN_k", knn_k), prob_knn),
  mkrow("CART",     prob_cart),
  mkrow("C5.0",     prob_c50),
  mkrow("RF_final", prob_rf),
  mkrow("GBM",      prob_gbm)
) %>%
  arrange(desc(AUC), desc(lift_top10), desc(recall_0.20))
leaderboard
#Model ranking (Test AUC). The logistic regression performs best (AUC ≈ 0.757), closely #followed by Random Forest (0.747) and CART (0.737). k-NN trails (0.713), and the C5.0 tree is #the weakest (0.663). The gap between Logit and RF is small (≈0.01), so both capture similar #signal; ensembles don’t dominate here, likely because the main effects (engagement/value) #are strong and fairly linear.
#Interpretation. An AUC around 0.75 means the top-scored positives are meaningfully #separable from negatives: the models rank likely buyers well enough to support targeted #marketing. C5.0’s lower AUC is consistent with our earlier results—single boosted tree with #imbalance handling via costs/weights helped recall but didn’t yield strong ranking power on #test data.
#Business takeaway. If we want the most interpretable and stable choice, pick Logit as the #primary model (best AUC, transparent coefficients, easy to operationalize). If we prefer a #slightly more flexible model with comparable performance, RF is a solid runner-up. In both #cases, use probability thresholds (e.g., 0.2–0.4) or decile targeting rather than a fixed 0.5 #cutoff to manage the class imbalance and marketing capacity. Report top-decile lift and #expected incremental conversions at the chosen contact depth to translate these AUC #differences into ROI terms
