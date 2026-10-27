from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
DATA_FILE = DATA_DIR / "Online Retail.xlsx"
# UCI's public dataset archive. If this URL changes, download from the official page
# and place Online Retail.xlsx in the data/ folder.
DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
SEED = 558
ALPHA = 0.05


def ensure_dirs():
    for p in (DATA_DIR, OUT_DIR, REPORT_DIR):
        p.mkdir(parents=True, exist_ok=True)


def download_dataset():
    if DATA_FILE.exists():
        print(f"Using existing dataset: {DATA_FILE}")
        return
    zip_path = DATA_DIR / "online-retail.zip"
    print("Downloading UCI Online Retail dataset (~22.6 MB)...")
    try:
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            candidates = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))]
            if not candidates:
                raise RuntimeError("The downloaded archive did not contain an Excel file.")
            chosen = candidates[0]
            with zf.open(chosen) as src, open(DATA_FILE, "wb") as dst:
                dst.write(src.read())
        print(f"Dataset saved to {DATA_FILE}")
    except Exception as exc:
        raise RuntimeError(
            "Automatic download failed. Download 'Online Retail.xlsx' from "
            "https://archive.ics.uci.edu/dataset/352/online+retail and put it in data/."
        ) from exc


def load_data():
    df = pd.read_excel(DATA_FILE, engine="openpyxl")
    expected = {"InvoiceNo", "StockCode", "Description", "Quantity",
                "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["Country"] = df["Country"].astype("string").fillna("Unknown")
    df["Description"] = df["Description"].astype("string")
    df["line_value"] = df["Quantity"] * df["UnitPrice"]
    return df



def write_raw_data_dictionary(raw):
    """Write type, unit, observed range/categories, and missingness for each original field."""
    units = {
        "InvoiceNo": "invoice identifier", "StockCode": "product identifier",
        "Description": "text", "Quantity": "units per line",
        "InvoiceDate": "timestamp", "UnitPrice": "GBP per unit",
        "CustomerID": "customer identifier", "Country": "country label",
    }
    nominal = {"InvoiceNo", "StockCode", "Description", "CustomerID", "Country"}
    rows = []
    for col in raw.columns:
        s = raw[col]
        if col in nominal:
            kind = "nominal/text"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        elif pd.api.types.is_integer_dtype(s):
            kind = "discrete"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "continuous"
        else:
            kind = "nominal/text"
        if pd.api.types.is_numeric_dtype(s):
            observed_range = f"{s.min()} to {s.max()}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            observed_range = f"{s.min()} to {s.max()}"
        else:
            observed_range = f"{s.nunique(dropna=True)} distinct observed values"
        rows.append({
            "feature": col, "type": kind, "unit": units.get(col, "see description"),
            "observed_range_or_levels": observed_range,
            "missing_n": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 4),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "raw_data_dictionary.csv", index=False)

def prepare_data(raw):
    # Do not impute customer IDs: a fabricated ID would create false customer histories.
    x = raw.copy()
    x["is_cancellation"] = x["InvoiceNo"].str.upper().str.startswith("C")
    audit = {
        "raw_rows": int(len(x)),
        "raw_columns": int(raw.shape[1]),
        "missing_customer_id_rows": int(x["CustomerID"].isna().sum()),
        "missing_description_rows": int(x["Description"].isna().sum()),
        "exact_duplicate_rows": int(x.duplicated().sum()),
        "cancellation_invoice_rows": int(x["is_cancellation"].sum()),
        "nonpositive_quantity_rows": int((x["Quantity"] <= 0).sum()),
        "nonpositive_price_rows": int((x["UnitPrice"] <= 0).sum()),
        "raw_unique_customers_with_id": int(x["CustomerID"].nunique(dropna=True)),
        "raw_unique_invoices": int(x["InvoiceNo"].nunique()),
        "date_min": str(x["InvoiceDate"].min()),
        "date_max": str(x["InvoiceDate"].max()),
    }

    # Analysis population: identified customers and positive sale lines, excluding
    # cancellation invoices, nonpositive quantities/prices, invalid dates and exact duplicates.
    clean = x.drop_duplicates().copy()
    clean = clean[
        clean["CustomerID"].notna()
        & clean["InvoiceDate"].notna()
        & (~clean["is_cancellation"])
        & (clean["Quantity"] > 0)
        & (clean["UnitPrice"] > 0)
    ].copy()
    clean["CustomerID"] = clean["CustomerID"].astype("int64").astype(str)
    clean["invoice_value"] = clean["Quantity"] * clean["UnitPrice"]
    clean["InvoiceNo"] = clean["InvoiceNo"].astype(str)
    clean = clean[clean["invoice_value"].notna() & np.isfinite(clean["invoice_value"])]

    audit.update({
        "prepared_sale_line_rows": int(len(clean)),
        "prepared_columns_including_derived": int(clean.shape[1]),
        "prepared_unique_customers": int(clean["CustomerID"].nunique()),
        "prepared_unique_invoices": int(clean["InvoiceNo"].nunique()),
        "prepared_total_invoice_value_gbp": float(clean["invoice_value"].sum()),
        "encoded_size_example_country_38_levels": int(len(raw) * (raw.shape[1] - 1 + raw["Country"].nunique())),
    })
    return clean, audit


def customer_tables(clean):
    # Collapse item lines into invoice-level purchases.
    invoices = (clean.groupby(["CustomerID", "InvoiceNo"], as_index=False)
                .agg(invoice_date=("InvoiceDate", "min"),
                     invoice_value=("invoice_value", "sum"),
                     items=("Quantity", "sum"),
                     unique_products=("StockCode", "nunique"),
                     country=("Country", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])))
    invoices = invoices.sort_values(["CustomerID", "invoice_date", "InvoiceNo"])
    cust = (invoices.groupby("CustomerID")
            .agg(invoice_count=("InvoiceNo", "nunique"),
                 total_spend=("invoice_value", "sum"),
                 avg_invoice_value=("invoice_value", "mean"),
                 median_invoice_value=("invoice_value", "median"),
                 total_items=("items", "sum"),
                 product_diversity=("unique_products", "sum"),
                 first_purchase=("invoice_date", "min"),
                 last_purchase=("invoice_date", "max"),
                 country=("country", "first"))
            .reset_index())
    cust["repeat_customer"] = (cust["invoice_count"] >= 2).astype(int)
    cust["days_between_first_last"] = (cust["last_purchase"] - cust["first_purchase"]).dt.days
    cust["log_total_spend"] = np.log1p(cust["total_spend"].clip(lower=0))
    return invoices, cust


def confidence_interval_diff(a, b, confidence=0.95):
    # Welch CI for mean(a)-mean(b)
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return [None, None], None
    diff = float(np.mean(a) - np.mean(b))
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    )
    tcrit = stats.t.ppf(1 - (1-confidence)/2, df)
    return [float(diff - tcrit*se), float(diff + tcrit*se)], float(df)


def run_tests(invoices, customers):
    results = []
    # T1: customer-level first invoice value split at median; repeat means >=2 invoices
    first_invoice = (invoices.sort_values("invoice_date")
                     .groupby("CustomerID", as_index=False).first()[["CustomerID", "invoice_value"]]
                     .rename(columns={"invoice_value": "first_invoice_value"}))
    t1 = customers.merge(first_invoice, on="CustomerID", how="inner")
    cutoff = float(t1["first_invoice_value"].median())
    low = t1[t1["first_invoice_value"] <= cutoff]
    high = t1[t1["first_invoice_value"] > cutoff]
    counts = np.array([int(high["repeat_customer"].sum()), int(low["repeat_customer"].sum())])
    nobs = np.array([len(high), len(low)])
    z, p = proportions_ztest(count=counts, nobs=nobs, alternative="two-sided")
    p_high, p_low = counts[0]/nobs[0], counts[1]/nobs[1]
    diff = p_high-p_low
    se = np.sqrt(p_high*(1-p_high)/nobs[0] + p_low*(1-p_low)/nobs[1])
    ci = [float(diff-1.96*se), float(diff+1.96*se)]
    results.append({
        "test": "T1: Repeat-purchase rate by first-invoice spend",
        "h0": "Repeat-purchase proportions are equal for above-median and at/below-median first-invoice spend.",
        "h1": "Repeat-purchase proportions differ between the two groups.",
        "test_used": "Two-proportion z-test, two-sided",
        "alpha": ALPHA, "n_group_a": int(len(high)), "n_group_b": int(len(low)),
        "estimate_group_a": float(p_high), "estimate_group_b": float(p_low),
        "estimate_difference_a_minus_b": float(diff), "ci95_difference": ci,
        "statistic": float(z), "p_value": float(p),
        "decision": "Reject H0" if p < ALPHA else "Fail to reject H0",
        "practical_note": "Interpret the percentage-point difference, not only the p-value."
    })

    # T2: mean invoice value for repeat vs one-time customers; use invoice observations
    rep_ids = set(customers.loc[customers["repeat_customer"] == 1, "CustomerID"])
    rep_inv = invoices.loc[invoices["CustomerID"].isin(rep_ids), "invoice_value"]
    once_ids = set(customers.loc[customers["repeat_customer"] == 0, "CustomerID"])
    once_inv = invoices.loc[invoices["CustomerID"].isin(once_ids), "invoice_value"]
    # These invoice observations are clustered by customer; report this limitation.
    welch = stats.ttest_ind(rep_inv, once_inv, equal_var=False, alternative="two-sided", nan_policy="omit")
    ci2, df2 = confidence_interval_diff(rep_inv, once_inv)
    results.append({
        "test": "T2: Average invoice value for repeat vs one-time customers",
        "h0": "Mean invoice values are equal in the two customer groups.",
        "h1": "Mean invoice values differ in the two customer groups.",
        "test_used": "Welch two-sample t-test, two-sided",
        "alpha": ALPHA, "n_group_a": int(rep_inv.notna().sum()), "n_group_b": int(once_inv.notna().sum()),
        "estimate_group_a": float(rep_inv.mean()), "estimate_group_b": float(once_inv.mean()),
        "estimate_difference_a_minus_b": float(rep_inv.mean()-once_inv.mean()),
        "ci95_difference": ci2, "statistic": float(welch.statistic),
        "degrees_freedom_approx": df2, "p_value": float(welch.pvalue),
        "decision": "Reject H0" if welch.pvalue < ALPHA else "Fail to reject H0",
        "practical_note": "Invoice rows from repeat customers are clustered; treat this as exploratory and confirm with customer-level summaries or cluster-aware methods."
    })

    # T3: repeat rate UK vs all non-UK customers
    uk = customers[customers["country"] == "United Kingdom"]
    nonuk = customers[customers["country"] != "United Kingdom"]
    cnt = np.array([int(uk["repeat_customer"].sum()), int(nonuk["repeat_customer"].sum())])
    nn = np.array([len(uk), len(nonuk)])
    z3, p3 = proportions_ztest(count=cnt, nobs=nn, alternative="two-sided")
    r_uk, r_non = cnt[0]/nn[0], cnt[1]/nn[1]
    d3 = r_uk-r_non
    se3 = np.sqrt(r_uk*(1-r_uk)/nn[0] + r_non*(1-r_non)/nn[1])
    results.append({
        "test": "T3: Repeat-purchase rate, UK vs non-UK customers",
        "h0": "Repeat-purchase proportions are equal for UK and non-UK customers.",
        "h1": "Repeat-purchase proportions differ for UK and non-UK customers.",
        "test_used": "Two-proportion z-test, two-sided",
        "alpha": ALPHA, "n_group_a": int(len(uk)), "n_group_b": int(len(nonuk)),
        "estimate_group_a": float(r_uk), "estimate_group_b": float(r_non),
        "estimate_difference_a_minus_b": float(d3),
        "ci95_difference": [float(d3-1.96*se3), float(d3+1.96*se3)],
        "statistic": float(z3), "p_value": float(p3),
        "decision": "Reject H0" if p3 < ALPHA else "Fail to reject H0",
        "practical_note": "Non-UK includes many countries and may be heterogeneous; this comparison is descriptive, not causal."
    })
    return results


def make_ml_table(clean):
    # Fixed temporal design: history up to 2011-09-01; label purchase in next 90 days.
    cutoff = pd.Timestamp("2011-09-01")
    horizon_end = cutoff + pd.Timedelta(days=90)
    prior = clean[(clean["InvoiceDate"] < cutoff)].copy()
    future = clean[(clean["InvoiceDate"] >= cutoff) & (clean["InvoiceDate"] < horizon_end)].copy()
    if prior.empty or future.empty:
        raise ValueError("Not enough data before/after the fixed temporal cutoff for ML.")
    hist = (prior.groupby("CustomerID")
            .agg(hist_invoice_count=("InvoiceNo", "nunique"),
                 hist_spend=("invoice_value", "sum"),
                 hist_avg_invoice=("invoice_value", lambda s: float(s.sum()) / max(prior.loc[s.index, "InvoiceNo"].nunique(), 1)),
                 hist_items=("Quantity", "sum"),
                 hist_unique_products=("StockCode", "nunique"),
                 hist_last_purchase=("InvoiceDate", "max"),
                 country=("Country", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]))
            .reset_index())
    last_date = prior["InvoiceDate"].max()
    hist["recency_days"] = (last_date - hist["hist_last_purchase"]).dt.days
    future_ids = set(future["CustomerID"].unique())
    hist["repeat_next_90d"] = hist["CustomerID"].isin(future_ids).astype(int)
    hist["log_hist_spend"] = np.log1p(hist["hist_spend"].clip(lower=0))
    hist = hist.drop(columns=["hist_last_purchase"])
    # Only customers observed before cutoff are eligible; features use no future information.
    return hist, {"history_start": str(prior["InvoiceDate"].min()),
                  "history_end_exclusive": str(cutoff),
                  "label_window_start": str(cutoff),
                  "label_window_end_exclusive": str(horizon_end),
                  "history_customers": int(len(hist)),
                  "positive_repeat_labels": int(hist["repeat_next_90d"].sum()),
                  "positive_rate": float(hist["repeat_next_90d"].mean())}


def train_models(ml):
    target = "repeat_next_90d"
    features = ["hist_invoice_count", "hist_spend", "hist_avg_invoice",
                "hist_items", "hist_unique_products", "recency_days", "log_hist_spend", "country"]
    X, y = ml[features], ml[target]
    # Reproducible stratified customer-level holdout. Temporal label window already prevents leakage.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    numeric = [c for c in features if c != "country"]
    categorical = ["country"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "shallow_decision_tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                                         class_weight="balanced", random_state=SEED)
    }
    metrics = []
    predictions = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        pred = pipe.predict(X_test)
        try:
            prob = pipe.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else None
        except Exception:
            auc = None
        metrics.append({
            "model": name, "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "features_before_encoding": int(X.shape[1]), "fit_seconds": float(fit_seconds),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()
        })
        predictions[name] = (pipe, pred)
    return metrics


def make_plots(clean, invoices, customers, ml):
    plt.style.use("default")
    # 1: customer invoice count distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = customers["invoice_count"].clip(upper=10)
    counts.value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set(title="Customer purchase frequency (10 means 10 or more invoices)",
           xlabel="Number of invoices", ylabel="Number of customers")
    fig.tight_layout(); fig.savefig(OUT_DIR / "purchase_frequency.png", dpi=160); plt.close(fig)

    # 2: spend by repeat group, capped visually at 99th percentile
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df = customers.copy()
    cap = plot_df["total_spend"].quantile(.99)
    plot_df["spend_for_plot"] = plot_df["total_spend"].clip(upper=cap)
    plot_df.boxplot(column="spend_for_plot", by="repeat_customer", ax=ax, showfliers=False)
    ax.set_title("Customer spending by repeat-purchase status")
    ax.set_xlabel("Repeat customer (1 = at least two invoices)")
    ax.set_ylabel("Total spend (£), capped visually at 99th percentile")
    fig.suptitle("")
    fig.tight_layout(); fig.savefig(OUT_DIR / "spend_by_repeat.png", dpi=160); plt.close(fig)

    # 3: monthly repeat activity, based on distinct invoices
    monthly = invoices.assign(month=invoices["invoice_date"].dt.to_period("M").astype(str))
    m = monthly.groupby("month")["InvoiceNo"].nunique()
    fig, ax = plt.subplots(figsize=(7, 4))
    m.plot(kind="line", marker="o", ax=ax)
    ax.set(title="Monthly distinct invoices", xlabel="Month", ylabel="Distinct invoices")
    fig.tight_layout(); fig.savefig(OUT_DIR / "monthly_invoices.png", dpi=160); plt.close(fig)

    # 4: label balance for ML
    fig, ax = plt.subplots(figsize=(6, 4))
    ml["repeat_next_90d"].value_counts().sort_index().rename(index={0:"No",1:"Yes"}).plot(kind="bar", ax=ax)
    ax.set(title="ML target balance: purchase in next 90 days",
           xlabel="Future purchase", ylabel="Customers")
    fig.tight_layout(); fig.savefig(OUT_DIR / "ml_target_balance.png", dpi=160); plt.close(fig)


def main():
    ensure_dirs()
    download_dataset()
    raw = load_data()
    write_raw_data_dictionary(raw)
    clean, audit = prepare_data(raw)
    invoices, customers = customer_tables(clean)
    tests = run_tests(invoices, customers)
    ml, ml_info = make_ml_table(clean)
    model_metrics = train_models(ml)
    make_plots(clean, invoices, customers, ml)

    # Save tables for reproducibility and report writing.
    clean.to_csv(OUT_DIR / "prepared_transactions.csv", index=False)
    invoices.to_csv(OUT_DIR / "invoice_level.csv", index=False)
    customers.to_csv(OUT_DIR / "customer_level.csv", index=False)
    ml.to_csv(OUT_DIR / "ml_customer_features.csv", index=False)
    pd.DataFrame(tests).to_json(OUT_DIR / "hypothesis_tests.json", orient="records", indent=2)
    pd.DataFrame(model_metrics).to_json(OUT_DIR / "model_metrics.json", orient="records", indent=2)
    summary = {
        "dataset_source": "UCI Machine Learning Repository, Online Retail, DOI 10.24432/C5BW33",
        "audit": audit,
        "hypothesis_tests": tests,
        "ml_design": ml_info,
        "model_metrics": model_metrics,
        "notes": [
            "Do not interpret associations as causal effects.",
            "T2 invoice-level observations are clustered within customers; use a customer-level or cluster-aware sensitivity analysis before final submission.",
            "The ML label is purchase in the 90 days after 2011-09-01; customers without prior history are outside this prediction cohort.",
            "Use only observed, actual run outputs in the report; do not invent results."
        ]
    }
    (OUT_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== DATA AUDIT ===")
    print(json.dumps(audit, indent=2))
    print("\n=== HYPOTHESIS TESTS ===")
    print(pd.DataFrame(tests)[["test", "statistic", "p_value", "decision"]].to_string(index=False))
    print("\n=== ML METRICS ===")
    print(pd.DataFrame(model_metrics)[["model", "fit_seconds", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print(f"\nOutputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
