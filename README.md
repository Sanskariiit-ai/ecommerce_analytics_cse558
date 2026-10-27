# E-commerce Analytics — CSE558 Data Science

**Focus:** Customer behaviour and repeat purchases  
**Dataset:** UCI Online Retail (Chen, 2015), CC BY 4.0  
**Official source:** https://archive.ics.uci.edu/dataset/352/online+retail  
**DOI:** https://doi.org/10.24432/C5BW33

## Setup (Windows / VS Code)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\run_analysis.py
python src\confirmation_analysis.py
```

The first script downloads/loads the original UCI workbook and creates cleaned tables, customer features, EDA, hypothesis-test outputs and baseline ML metrics. If automatic download fails, download `Online Retail.xlsx` from the official UCI page and place it at `data\Online Retail.xlsx`.

## Key outputs

- `outputs/results_summary.json`: data audit, baseline tests and ML results
- `outputs/prepared_transactions.csv`: prepared transaction lines
- `outputs/invoice_level.csv`: one row per invoice
- `outputs/customer_level.csv`: one row per customer
- `outputs/ml_customer_features.csv`: 90-day prediction cohort
- `outputs/raw_data_dictionary.csv`: original-field type, units, observed range/levels and missingness (created after the raw workbook is loaded)\n- `outputs/data_dictionary_customer_level.csv`: customer-table data dictionary
- `outputs/exploration_customers.csv` and `outputs/confirmation_customers.csv`: split, seed 558
- `outputs/confirmation_hypothesis_tests.json`: three customer-level confirmation tests
- `outputs/scalability_experiment.csv`: fixed-test-set 25%, 50%, 100% training-size comparison
- `outputs/eda_*_exploration.png`: EDA figures made from the exploration customers
- `report/CSE558_Ecommerce_Analytics_Midterm_Report.tex` and `.pdf`: report based on the supplied course LaTeX template

## Dataset size check

The original workbook has 541,909 rows and 8 original columns. Country one-hot encoding with 38 observed levels gives `7 + 38 = 45` columns and `541,909 × 45 = 24,385,905` cells for original attributes only. The loader adds a derived line-value feature before its audit, so the audit calculation uses `8 + 38 = 46` columns and `541,909 × 46 = 24,927,814` cells. The exact Country cardinality is calculated at runtime; rerun `src/run_analysis.py` against the source workbook to verify it. The raw workbook is not bundled because it should be obtained from the cited official source.

## Statistical inference

The report uses a customer-level 80/20 exploration/confirmation split (seed 558). EDA figures use the exploration customers; `confirmation_analysis.py` recreates three tests on the held-out confirmation set. Two proportion tests report z statistics (degrees of freedom are not applicable); the continuous-outcome comparison uses Welch's t-test and its approximate degrees of freedom. Practical thresholds in the report are interpretive thresholds chosen for this write-up, not preregistered thresholds.

## ML and scale experiment

Predict purchase in the 90 days after 1 September 2011 using only pre-cutoff features. Logistic regression and a depth-4 decision tree are evaluated on the same fixed 830-customer test set, with training sizes of 621, 1,243 and 2,487. Runtime is machine-specific; this small dataset cannot establish production-scale scalability.

## Before submission

1. Replace all student/group metadata placeholders.
2. Publish code and README to a public repository and replace the code URL in the report.
3. Download the raw workbook from UCI and run both scripts to verify the exact encoded shape and reproduce outputs.
4. Review the report PDF (4 pages; limit is 8) and submit it with the script and README as required.
5. Treat findings as observational associations, not causal effects.

## Citation

Chen, D. (2015). *Online Retail*. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. CC BY 4.0.
