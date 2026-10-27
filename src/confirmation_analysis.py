"""Reproduce the exploration/confirmation split and three confirmation tests.
Requires outputs/customer_level.csv and outputs/prepared_transactions.csv from run_analysis.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 558

customers = pd.read_csv(OUT / "customer_level.csv")
transactions = pd.read_csv(OUT / "prepared_transactions.csv", parse_dates=["InvoiceDate"])
transactions["line_value"] = transactions["Quantity"] * transactions["UnitPrice"]
invoices = transactions.groupby(
    ["CustomerID", "InvoiceNo", "InvoiceDate", "Country"], as_index=False
).agg(invoice_value=("line_value", "sum"))
first = invoices.sort_values(["CustomerID", "InvoiceDate", "InvoiceNo"]).groupby(
    "CustomerID", as_index=False
).first()[["CustomerID", "invoice_value"]].rename(
    columns={"invoice_value": "first_invoice_value"}
)
customers = customers.merge(first, on="CustomerID", how="left")
explore, confirm = train_test_split(
    customers, test_size=0.20, random_state=SEED,
    stratify=customers["repeat_customer"]
)
explore.to_csv(OUT / "exploration_customers.csv", index=False)
confirm.to_csv(OUT / "confirmation_customers.csv", index=False)
threshold = float(explore["first_invoice_value"].median())
confirm = confirm.copy()
confirm["first_spend_group"] = np.where(
    confirm["first_invoice_value"] > threshold, "above_median", "at_or_below_median"
)

def proportion_test(group1, group0, name):
    x1, n1 = int(group1.sum()), len(group1)
    x0, n0 = int(group0.sum()), len(group0)
    p1, p0 = x1 / n1, x0 / n0
    difference = p1 - p0
    pooled = (x1 + x0) / (n1 + n0)
    se0 = np.sqrt(pooled * (1 - pooled) * (1/n1 + 1/n0))
    z = difference / se0
    p_value = 2 * stats.norm.sf(abs(z))
    se_ci = np.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)
    ci = [difference - 1.96*se_ci, difference + 1.96*se_ci]
    return {"test": name, "n1": n1, "successes1": x1, "rate1": p1,
            "n0": n0, "successes0": x0, "rate0": p0,
            "difference": difference, "ci95": ci, "z": z, "p_value": p_value,
            "degrees_of_freedom": None}

g1 = confirm.loc[confirm.first_spend_group == "above_median", "repeat_customer"]
g0 = confirm.loc[confirm.first_spend_group == "at_or_below_median", "repeat_customer"]
t1 = proportion_test(g1, g0, "T1: Repeat purchase by first invoice spend")

repeat = confirm.loc[confirm.repeat_customer == 1, "avg_invoice_value"].dropna()
one_time = confirm.loc[confirm.repeat_customer == 0, "avg_invoice_value"].dropna()
welch = stats.ttest_ind(repeat, one_time, equal_var=False)
difference = float(repeat.mean() - one_time.mean())
se = np.sqrt(repeat.var(ddof=1)/len(repeat) + one_time.var(ddof=1)/len(one_time))
df = (repeat.var(ddof=1)/len(repeat) + one_time.var(ddof=1)/len(one_time))**2 / (
    (repeat.var(ddof=1)/len(repeat))**2/(len(repeat)-1) +
    (one_time.var(ddof=1)/len(one_time))**2/(len(one_time)-1)
)
ci = [difference - stats.t.ppf(.975, df)*se, difference + stats.t.ppf(.975, df)*se]
t2 = {"test": "T2: Customer-level average invoice value by repeat status",
      "n_repeat": len(repeat), "n_one_time": len(one_time),
      "mean_repeat": float(repeat.mean()), "mean_one_time": float(one_time.mean()),
      "difference": difference, "ci95": ci, "t": float(welch.statistic),
      "df": float(df), "p_value": float(welch.pvalue)}

uk = confirm.loc[confirm.country == "United Kingdom", "repeat_customer"]
nonuk = confirm.loc[confirm.country != "United Kingdom", "repeat_customer"]
t3 = proportion_test(uk, nonuk, "T3: Repeat purchase UK vs non-UK")

result = {"seed": SEED, "exploration_n": len(explore), "confirmation_n": len(confirm),
          "exploration_median_first_invoice_value_gbp": threshold,
          "tests": [t1, t2, t3]}
(OUT / "confirmation_hypothesis_tests.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
