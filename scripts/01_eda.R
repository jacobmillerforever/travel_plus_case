# ----------------------------------------------------------------------------
# Assignment 1 (IDS 572) - TravelPlus RoadsidePlus Case Study
# Contributor: Jacob Miller - Exploratory Data Analysis Lead
#
# Objective:
#   Outline all exploratory analyses required to understand the dataset,
#   characterize purchase behavior, and evaluate signal strength of different
#   predictor groups before any modeling begins.
#
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
library(yardstick)


df <- readr::read_csv("data/dataTravelPlus.csv")

df <- df %>%
  mutate(
    Purchase = factor(Purchase, levels = c(0, 1)),
    Purchase_num = as.integer(Purchase) - 1L   
  )

skim(df)

# 2) Compute baseline metrics
#   - Calculate the overall Purchase rate (Yes vs. No) for the contacted group.
n_total   <- nrow(df)
n_buyers  <- sum(df$Purchase_num)
buy_rate  <- mean(df$Purchase_num) * 100

cat(
  sprintf(
    "Of the %d people contacted, %d (%.1f%%) purchased the add-on travel insurance.\n",
    n_total, n_buyers, buy_rate
  )
)

# 3) Explore numeric predictors versus Purchase

num_vars <- df %>% select(where(is.numeric)) %>% names()
num_vars <- setdiff(num_vars, c("Purchase_num"))
cat_vars <- df %>% select(where(~ is.character(.x) || is.factor(.x))) %>% names()
cat_vars <- setdiff(cat_vars, "Purchase")    



#Summary stats(mean, min, max) by Purchase across all numeric variables
num_summary <- df %>%
  group_by(Purchase) %>%
  summarise(
    across(all_of(num_vars), list(mean = mean, min = min, max = max),
           .names = "{.col}_{.fn}"),
    .groups = "drop"
  )

for (v in num_vars) {
  test  <- t.test(df[[v]] ~ df$Purchase)
  t_val <- round(test$statistic, 2)
  dfree <- round(test$parameter, 1)
  pval  <- test$p.value
  p_lab <- ifelse(pval < 0.001, "< 0.001", sprintf("= %.3f", pval))
  
  p <- ggplot(df, aes(x = Purchase, y = .data[[v]], fill = Purchase)) +
    geom_boxplot() +
    labs(
      title    = paste("Boxplot:", v, "by Purchase"),
      subtitle = paste0("t = ", t_val, ", df = ", dfree, ", p ", p_lab),
      x = "Purchase", y = v
    ) +
    theme_minimal() + theme(legend.position = "none")
  print(p)
}

one_var_auc <- function(v) {
  f  <- as.formula(paste("Purchase ~", v))
  mf <- model.frame(f, data = df)                 # drops rows with NA in either col
  fit <- glm(f, data = mf, family = binomial)
  p   <- fitted(fit)
  auc <- roc_auc_vec(truth = mf$Purchase, estimate = p, event_level = "second")
  tibble(var = v, auc = auc)
}

results_num <- purrr::map_dfr(num_vars, one_var_auc) %>% arrange(desc(auc))
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
chi_results <- data.frame(variable = character(), df = numeric(),
                          X2 = numeric(), p_value = numeric(), stringsAsFactors = FALSE)

for (v in cat_vars) {
  tab <- table(df[[v]], df$Purchase_num)
  chi <- chisq.test(tab)
  chi_results <- rbind(
    chi_results,
    data.frame(
      variable = v,
      df = chi$parameter,
      X2 = chi$statistic,
      p_value = chi$p.value
    )
  )
}


for (v in cat_vars) {
  tab <- with(df, tapply(Purchase_num, df[[v]], mean, na.rm = TRUE))
  
  chi_row <- chi_results[chi_results$variable == v, ]
  x2   <- if (nrow(chi_row)) round(chi_row$X2, 2) else NA_real_
  pval <- if (nrow(chi_row)) chi_row$p_value else NA_real_
  p_lab <- if (is.na(pval)) "NA" else if (pval < 0.001) "< 0.001" else sprintf("= %.3f", pval)
  
  ymax <- max(tab, na.rm = TRUE); if (!is.finite(ymax)) next
  
  barplot(tab,
          main = paste("Purchase Rate by", v),
          sub  = bquote(chi^2 == .(x2) ~ ", p " ~ .(p_lab)),
          ylab = "Purchase Rate", xlab = v,
          ylim = c(0, ymax * 1.1), col = "skyblue")
  
  text(x = seq_along(tab), y = tab,
       labels = paste0(round(100 * tab, 1), "%"),
       pos = 3, cex = 0.8)
}

results_cat <- purrr::map_dfr(cat_vars, one_var_auc) %>% arrange(desc(auc))
print(results_cat)

results_all <- dplyr::bind_rows(results_num, results_cat) %>% arrange(desc(auc))
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

