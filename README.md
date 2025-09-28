# TravelPlus RoadsidePlus Case Study

This repository contains the scaffold for completing the IDS 572 Assignment 1 case study on targeting customers for TravelPlus Insurance's RoadsidePlus add-on. The workflow is organized into three R scripts that correspond to the analytical deliverables outlined in the assignment brief (`travel_plus_case_assignment.md`).

## Repository Structure and Deliverables

| Script | Focus Area | Key Deliverables Covered |
| --- | --- | --- |
| `scripts/01_eda.R` | Exploratory data analysis | Overall purchase rate confirmation, numeric and categorical predictor diagnostics, lifestyle feature assessment, composite score evaluation, and synthesized insights for modeling handoff. |
| `scripts/02_data_preparation_and_modeling.R` | Data preparation and predictive modeling | Preprocessing pipeline documentation, data partition strategy, tuned models (logistic regression, k-NN, decision tree, random forest, GBM), model evaluation metrics, and cross-model variable importance comparisons. |
| `scripts/03_marketing_evaluation_and_recommendation.R` | Marketing impact analysis and recommendation | Lift tables/visualizations by decile, targeting scenario comparisons (top 10/20/30%), incremental lift methodology, and executive-ready marketing recommendations. |

Supporting assets include the `data/` directory for provided datasets and `markdown/` for any narrative reporting you choose to add. The `renv.lock` file captures the expected R package environment for reproducibility.

## Getting Started

Clone the repository and restore the R environment defined in `renv.lock`:

```bash
git clone https://github.com/jacobmillerforever/travel_plus_case.git
cd travel_plus_case
R -e "renv::restore()"
```

After the environment is restored, open the R project (`travel_plus_case.Rproj`) in RStudio or your preferred IDE and execute each script in sequence, supplying outputs (tables, plots, narratives) that fulfill the assignment requirements described above.

## Additional Notes

* `travel_plus_case_assignment.md` reproduces the full assignment prompt and should be referenced to ensure all analytical questions are addressed.
* The scripts are intentionally written as detailed checklists; implement the analysis directly in these files or source modular functions as needed.
* If you add new dependencies, update the `renv` environment (`renv::snapshot()`) to keep collaborators in sync.
