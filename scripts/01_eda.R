# ----------------------------------------------------------------------------
# Assignment 1 (IDS 572) - TravelPlus RoadsidePlus Case Study
# Contributor: Jacob Miller - Exploratory Data Analysis Lead
#
# Objective:
#   Outline all exploratory analyses required to understand the dataset,
#   characterize purchase behavior, and evaluate signal strength of different
#   predictor groups before any modeling begins.
#
# Data Inputs Needed:
#   - data/dataTravelPlus.csv (contacted customers only)
#   - Description of variables from travel_plus_case_assignment.md
#
# Key Deliverables for this Script:
#   1. Overall purchase rate for contacted customers.
#   2. Visual and tabular summaries of predictor relationships with Purchase.
#   3. Diagnostic commentary on lifestyle variables versus other feature groups.
#   4. Initial assessment of composite score usefulness.
#   5. Documentation of insights to hand off to modeling teammate.
#
# ---------------------------------------------------------------------------
# 1) Set up environment

library(tidyverse)
library(skimr)
library(broom)


df <- readr::read_csv("data/dataTravelPlus.csv")
skim(df)

# 2) Compute baseline metrics
#   - Calculate the overall Purchase rate (Yes vs. No) for the contacted group.

print(paste("Of the 20,000 people contacted", sum(df$Purchase), "or",sum(df['Purchase'])/20000 * 100,"%, purchased the add on travel insurance."))

# 3) Explore numeric predictors versus Purchase


num_vars <- names(df)[sapply(df, is.numeric)]
num_vars <- setdiff(num_vars, "Purchase")

for (v in num_vars) {
  boxplot(df[[v]] ~ df$Purchase,
          main = paste("Boxplot of", v, "by Purchase"),
          xlab = "Purchase",
          ylab = v,
          col = c("lightblue", "salmon"))
}



results_num <- data.frame(var = character(), r2 = numeric(), stringsAsFactors = FALSE)

for (v in num_vars) {
  f <- as.formula(paste("Purchase ~", v))
  fit <- glm(f, data = df, family = binomial)
  fit_null <- glm(Purchase ~ 1, data = df, family = binomial)
  r2 <- 1 - (as.numeric(logLik(fit)) / as.numeric(logLik(fit_null)))
  results_num <- rbind(results_num, data.frame(var = v, r2 = r2))
}

# Sort by R^2 descending
results_num <- results_num[order(-results_num$r2), ]
print(results_num)

# 4) Explore categorical predictors versus Purchase
#   - Identify categorical variables (Region, MaritalStatus, HomeOwner, etc.).
#   - For each categorical variable:
#       * Generate contingency tables (counts and purchase rates by category).
#       * Visualize purchase rates via bar charts with confidence intervals.
#       * Conduct chi-square tests (or Fisher exact where appropriate) to assess
#         whether category distributions differ significantly between purchasers
#         and non-purchasers.
#       * Annotate findings on categories that appear predictive vs. noisy.
cat_vars <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]

# bar plots of purchase rate
for (v in cat_vars) {
  # Compute purchase rate by category
  tab <- tapply(df$Purchase, df[[v]], mean, na.rm = TRUE)
  
  # Make barplot
  barplot(tab,
          main = paste("Purchase Rate by", v),
          ylab = "Purchase Rate",
          xlab = v,
          ylim = c(0, max(tab, na.rm = TRUE) * 1.1),
          col = "skyblue")
  
  # Add % labels above bars
  text(x = seq_along(tab),
       y = tab,
       labels = paste0(round(100*tab, 1), "%"),
       pos = 3, cex = 0.8)
}

for (v in cat_vars) {
  tab <- table(df$Purchase, df[[v]])
  tab
  print(v)
  print(chisq.test(tab))
}




results_cat <- data.frame(var = character(), r2 = numeric())

for (var in cat_vars) {
  form <- as.formula(paste("Purchase ~", var, "- 1"))
  fit <- glm(form, data = df, family = binomial)
  fit_null <- glm(Purchase ~ 1, data = df, family = binomial)
  r2_val <- 1 - (as.numeric(logLik(fit)) / as.numeric(logLik(fit_null)))
  results_cat <- rbind(results_cat, data.frame(var = var, r2 = r2_val))
}

print(results_cat)



results_all <- bind_rows(results_num, results_cat) %>%
  arrange(desc(r2))

print(results_all)

# 5) Evaluate lifestyle variables specifically
#   - Compile all lifestyle fields (StreamingHours, PetOwnership, CommuteDistance,
#     DiningOutFreq, AppDownloads) into focused visualizations and summaries.
#   - Compare their predictive signal to core engagement and spending variables
#     (e.g., overlay distributions, compute mutual information, etc.).
#   - Provide explicit commentary on whether the data supports Joi's intuition
#     that lifestyle variables may mostly add noise.




# 6) Examine composite scores
#   - For each score (Score_Engagement, Score_Value, Score_Risk, Score_Upsell):
#       * Plot score distributions stratified by Purchase (boxplot/violin/histogram).
#       * Compute mean/median by Purchase status to gauge directionality.
#   - Investigate redundancy:
#       * Correlate each composite score with its underlying raw components
#         (e.g., Score_Engagement vs. email/web metrics; Score_Upsell vs. prior add-ons).
#       * Document whether the composite adds distinct information beyond raw
#         metrics (e.g., via partial correlations or logistic models with/without
#         the composite score).
#   - Summarize which scores appear most promising to keep for modeling.
#
# 7) Synthesize findings
#   - Capture top predictive variables, unexpected patterns, and any data quality
#     concerns discovered during EDA.
#   - Draft bullet-point insights to pass to the modeling teammate highlighting
#     variables to prioritize, interactions worth testing, and any preprocessing
#     needs identified (e.g., skewed distributions requiring transformation).
#   - Save all plots/tables to an output folder (if needed) for inclusion in the
#     final report.

