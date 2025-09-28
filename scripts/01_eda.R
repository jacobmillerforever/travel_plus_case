# ----------------------------------------------------------------------------
# Assignment 1 (IDS 572) - TravelPlus RoadsidePlus Case Study
# Contributor: Analyst 1 - Exploratory Data Analysis Lead
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
#   - Load tidyverse (or preferred plotting/EDA libraries) for data wrangling,
#     visualization, and table creation.
#   - Import helper packages for summary statistics (e.g., skimr, janitor).
#   - Read the TravelPlus dataset and immediately inspect structure (str, glimpse)
#     to confirm data types match expectations from the case brief.
#   - Audit missing values globally and by column; decide whether to flag any
#     variables with high missingness for imputation or exclusion.
#
# 2) Compute baseline metrics
#   - Calculate the overall Purchase rate (Yes vs. No) for the contacted group.
#   - Produce a simple summary table showing counts and percentages for Purchase.
#   - Capture insights in comments for the business narrative (e.g., confirm the
#     ~8% response rate mentioned in the case description).
#
# 3) Explore numeric predictors versus Purchase
#   - Identify numeric columns (demographics, engagement, spending, scores, etc.).
#   - For each numeric variable:
#       * Create side-by-side visualizations (boxplots, density plots, violin plots)
#         comparing purchasers vs. non-purchasers.
#       * Compute group-level summary statistics (mean, median, standard deviation)
#         and capture differences in a comparison table.
#       * Flag variables that show noticeable separation between Purchase groups.
#   - Consider adding correlation analysis or standardized mean differences to
#     quantify effect sizes.
#
# 4) Explore categorical predictors versus Purchase
#   - Identify categorical variables (Region, MaritalStatus, HomeOwner, etc.).
#   - For each categorical variable:
#       * Generate contingency tables (counts and purchase rates by category).
#       * Visualize purchase rates via bar charts with confidence intervals.
#       * Conduct chi-square tests (or Fisher exact where appropriate) to assess
#         whether category distributions differ significantly between purchasers
#         and non-purchasers.
#       * Annotate findings on categories that appear predictive vs. noisy.
#
# 5) Evaluate lifestyle variables specifically
#   - Compile all lifestyle fields (StreamingHours, PetOwnership, CommuteDistance,
#     DiningOutFreq, AppDownloads) into focused visualizations and summaries.
#   - Compare their predictive signal to core engagement and spending variables
#     (e.g., overlay distributions, compute mutual information, etc.).
#   - Provide explicit commentary on whether the data supports Joi's intuition
#     that lifestyle variables may mostly add noise.
#
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
#
# NOTE: Do not implement modeling here—this script strictly documents exploratory
#       steps and observations for the case study handoff.
# ----------------------------------------------------------------------------
