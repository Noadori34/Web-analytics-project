"""
Phase 3: Salary Modeling Pipeline (03_salary_model.py)
----------------------------------------------------------
Meets all Phase 3 requirements:
  1. Joins exploded skills (Datasets/job_skills_clean.csv) with job metadata &
     salaries (Datasets/combined_data.csv) via `job_uid`.
  2. Cleans salary sample: drops missing salaries, converts hourly rates
     ($15-$300/hr * 40 * 52), filters to USD, excludes interns, trims 1%/99% tail
     outliers, and computes log(annual_salary).
  3. Fits Baseline (Skills-Only) and Controlled OLS regressions (HC3 robust SEs)
     controlling for seniority (`level`), role (`role_category`), remote/work
     mode (`work_mode`), and data source (`source`).
  4. Compares simple (unadjusted) salary differences vs. adjusted log-linear
     associations ((exp(beta) - 1) * 100%).
  5. Exports `salary_coefficients.csv`, `salary_model_results.csv`,
     `regression_summary.txt`, and 3 diagnostic/results figures.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
CONFIG = {
    "skills_file": Path("Datasets/job_skills_clean.csv"),
    "jobs_file": Path("Datasets/combined_data.csv"),
    "regression_dir": Path("salary_model_results/regression"),
    "figures_dir": Path("salary_model_results/figures"),
    # Sample filtering & outlier handling
    "target_currency": "USD",
    "hourly_min": 15.0,
    "hourly_max": 300.0,
    "min_salary_usd": 15_000,
    "max_salary_usd": 600_000,
    "winsorize_quantiles": (0.01, 0.99),
    "exclude_levels": ["Intern", "Internship"],
    # Feature engineering
    "min_skill_count": 20,
    "control_cols": ["level", "work_mode", "source", "role_category"],
    "alpha": 0.05,
}


# ==============================================================================
# 2. LOAD & JOIN PHASE 2 TABLES ON `job_uid`
# ==============================================================================
def load_merged_dataset(config: dict) -> tuple[pd.DataFrame, list[str]]:
    skills_long = pd.read_csv(config["skills_file"])
    jobs_df = pd.read_csv(config["jobs_file"])
    print(
        f"[1/6] Loaded {len(jobs_df):,} jobs from {config['jobs_file']} "
        f"and {len(skills_long):,} skill rows from {config['skills_file']}"
    )

    # Preserve distinct languages before removing punctuation.
    skill_labels = (
        skills_long["skill"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    skill_labels = skill_labels.replace({
        "c++": "cpp",
        "c#": "csharp",
    })

    skills_long["skill_clean"] = (
        skill_labels
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    skills_long = skills_long[skills_long["skill_clean"] != ""].drop_duplicates(
        subset=["job_uid", "skill_clean"]
    )

    skill_matrix = (
        pd.crosstab(skills_long["job_uid"], skills_long["skill_clean"])
        .clip(upper=1)
        .add_prefix("skill_")
        .reset_index()
    )

    merged = jobs_df.merge(skill_matrix, on="job_uid", how="left")
    all_skill_cols = [c for c in merged.columns if c.startswith("skill_")]
    merged[all_skill_cols] = merged[all_skill_cols].fillna(0).astype(int)

    print(f"      Merged dataset: {merged.shape[0]:,} rows x {merged.shape[1]} cols ({len(all_skill_cols)} raw skills)")
    return merged, all_skill_cols


# ==============================================================================
# 3. SALARY SAMPLE PREPARATION (MISSING, HOURLY, CURRENCY, OUTLIERS, LOG)
# ==============================================================================
MIXED_SALARY_RANGE = re.compile(
    r"(?:USD\s*|\$\s*)?"
    r"(?P<low>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:/|per\s+)\s*"
    r"(?P<low_period>hours?|hrs?|years?|yrs?|months?|mos?|weeks?|wks?)"
    r"\b\s*(?:to|[-–—])\s*"
    r"(?:USD\s*|\$\s*)?"
    r"(?P<high>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:/|per\s+)\s*"
    r"(?P<high_period>hours?|hrs?|years?|yrs?|months?|mos?|weeks?|wks?)"
    r"\b",
    flags=re.IGNORECASE,
)


def correct_mixed_period_salary(row):
    """
    Correct explicit mixed-period salary ranges in USD descriptions.

    Returns:
        annual salary, correction status

    Unrecognised formats retain the existing salary.
    Conflicting or invalid mixed-period ranges are excluded.
    """
    original_salary = row["annual_salary"]
    description = row.get("description", "")
    currency = str(row.get("currency", "")).strip().upper()

    if currency != "USD" or not isinstance(description, str):
        return original_salary, "unchanged"

    def multiplier(period):
        period = period.lower()

        if period.startswith(("hour", "hr")):
            return 2080
        if period.startswith(("week", "wk")):
            return 52
        if period.startswith(("month", "mo")):
            return 12
        return 1  # Pattern permits only year/yr alternatives here.

    annual_ranges = set()

    for match in MIXED_SALARY_RANGE.finditer(description):
        low_factor = multiplier(match.group("low_period"))
        high_factor = multiplier(match.group("high_period"))

        # Only intervene when the two bounds use different periods.
        if low_factor == high_factor:
            continue

        low = float(match.group("low").replace(",", ""))
        high = float(match.group("high").replace(",", ""))

        annual_low = low * low_factor
        annual_high = high * high_factor

        annual_ranges.add((annual_low, annual_high))

    if not annual_ranges:
        return original_salary, "unchanged"

    # Repeated identical ranges are fine; different ranges are ambiguous.
    if len(annual_ranges) != 1:
        return np.nan, "excluded_multiple_mixed_ranges"

    annual_low, annual_high = next(iter(annual_ranges))

    if annual_low <= 0 or annual_high < annual_low:
        return np.nan, "excluded_invalid_mixed_range"

    corrected_salary = round(
        (annual_low + annual_high) / 2,
        2,
    )

    return corrected_salary, "corrected_mixed_period"

def clean_salary_sample(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = df.copy()
    initial_n = len(data)

    # 1. Currency filter (USD only)
    if "currency" in data.columns and config["target_currency"]:
        data = data[data["currency"].astype(str).str.upper() == config["target_currency"]].copy()
    after_currency_n = len(data)

    # 2. Correct explicit mixed-period salaries before dropping
    # missing values or applying salary bounds.
    data["annual_salary"] = pd.to_numeric(
        data["annual_salary"],
        errors="coerce",
    )

    data["annual_salary_before_correction"] = data["annual_salary"]

    corrections = data.apply(
        correct_mixed_period_salary,
        axis=1,
        result_type="expand",
    )

    corrections.columns = [
        "annual_salary",
        "salary_correction_status",
    ]

    data["annual_salary"] = corrections["annual_salary"]
    data["salary_correction_status"] = corrections[
        "salary_correction_status"
    ]

    # Save a record of corrections and exclusions.
    changed = data["salary_correction_status"].ne("unchanged")

    audit_columns = [
        "job_uid",
        "title",
        "annual_salary_before_correction",
        "annual_salary",
        "salary_correction_status",
    ]

    audit = data.loc[changed, audit_columns].copy()

    config["regression_dir"].mkdir(parents=True, exist_ok=True)

    audit.to_csv(
        config["regression_dir"] / "salary_corrections.csv",
        index=False,
    )

    print("\nMixed-period salary handling:")
    print(data["salary_correction_status"].value_counts().to_string())

    # Drop missing or explicitly excluded salaries after correction.
    data = data.dropna(subset=["annual_salary"]).copy()
    after_missing_n = len(data)

    # 3. Convert any un-annualized hourly salaries (HourlySalary * 40 * 52)
    hourly_mask = (data["annual_salary"] >= config["hourly_min"]) & (
        data["annual_salary"] <= config["hourly_max"]
    )
    n_hourly_converted = int(hourly_mask.sum())
    data.loc[hourly_mask, "annual_salary"] = data.loc[hourly_mask, "annual_salary"] * 40 * 52

    # 4. Exclude internships
    if "level" in data.columns and config["exclude_levels"]:
        data = data[~data["level"].astype(str).isin(config["exclude_levels"])].copy()

    # 5. Economic bounds + 1st/99th percentile outlier trimming
    data = data[
        (data["annual_salary"] >= config["min_salary_usd"])
        & (data["annual_salary"] <= config["max_salary_usd"])
    ].copy()

    if config["winsorize_quantiles"] is not None:
        q_low, q_high = config["winsorize_quantiles"]
        low_val, high_val = data["annual_salary"].quantile([q_low, q_high])
        data = data[(data["annual_salary"] >= low_val) & (data["annual_salary"] <= high_val)].copy()

    # 6. Log transformation
    data["log_salary"] = np.log(data["annual_salary"])

    print(
        f"[2/6] Salary sample pipeline: Total={initial_n:,} -> USD={after_currency_n:,} -> "
        f"Non-missing={after_missing_n:,} (hourly converted={n_hourly_converted}) -> "
        f"Final clean={len(data):,}"
    )
    print(
        f"      Salary stats (USD): Median=${data['annual_salary'].median():,.0f} | "
        f"Mean=${data['annual_salary'].mean():,.0f} | "
        f"Min=${data['annual_salary'].min():,.0f} | Max=${data['annual_salary'].max():,.0f}"
    )
    return data


# ==============================================================================
# 4. FEATURE ENGINEERING (SKILLS + STATISTICAL CONTROLS)
# ==============================================================================
def prepare_features(
    df: pd.DataFrame, all_skill_cols: list[str], config: dict
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    data = df.copy()

    kept_skills = [c for c in all_skill_cols if data[c].sum() >= config["min_skill_count"]]
    print(
        f"[3/6] Skill features: kept {len(kept_skills)} skills (>= {config['min_skill_count']} postings), "
        f"dropped {len(all_skill_cols) - len(kept_skills)} rare skills"
    )

    t = (
        data["title"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[-_/]+", " ", regex=True)
    )

    # Specific roles are checked before broader engineering roles.
    bi_role = t.str.contains(
        r"\bbi\b|\bbusiness intelligence\b|\btableau\b|"
        r"\bpower\s*bi\b|\breporting analyst\b",
        regex=True,
        na=False,
    )

    science_role = t.str.contains(
        r"\bdata scien(?:ce|tist|tists)\b|"
        r"\bapplied scientist\b|\bresearch scientist\b|"
        r"\bmachine learning\b|\bdeep learning\b|"
        r"\bai\b|\bml\b|\bartificial intelligence\b|"
        r"\bcomputer vision\b|\bnlp\b",
        regex=True,
        na=False,
    )

    data_engineering_role = t.str.contains(
        r"\bdata engineer(?:ing)?\b|"
        r"\banalytics engineer(?:ing)?\b|"
        r"\bdata architect\b|\bdatabase\b|"
        r"\betl\b|\bdata pipeline\b|\bdbt\b",
        regex=True,
        na=False,
    )

    product_marketing_role = t.str.contains(
        r"\bproduct\b|\bgrowth\b|\bmarketing\b|"
        r"\bweb\b|\bdigital\b",
        regex=True,
        na=False,
    )

    analyst_role = t.str.contains(
        r"\bdata analyst\b|\bbusiness analyst\b|"
        r"\banalytics analyst\b",
        regex=True,
        na=False,
    )

    other_engineering_role = t.str.contains(
        r"\bengineer(?:ing)?\b|\bdeveloper\b|\barchitect\b",
        regex=True,
        na=False,
    )

    data["role_category"] = np.select(
        [
            bi_role,
            science_role,
            data_engineering_role,
            product_marketing_role,
            analyst_role,
            other_engineering_role,
        ],
        [
            "BI / Reporting",
            "Data Science / ML",
            "Data / Analytics Eng",
            "Web / Marketing / Product",
            "General Data Analyst",
            "Other Engineering",
        ],
        default="Other / Unclassified",
    )

    active_controls = []
    for col in config["control_cols"]:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
            if data[col].nunique() > 1:
                active_controls.append(col)

    control_dummies = pd.get_dummies(data[active_controls], drop_first=True, dtype=int)
    print(f"      Controls included: {active_controls} -> {control_dummies.shape[1]} dummy columns")
    return data, kept_skills, control_dummies


# ==============================================================================
# 5. OLS REGRESSION ANALYSIS (UNADJUSTED VS. ADJUSTED ASSOCIATIONS)
# ==============================================================================
def run_regression_analysis(
    df: pd.DataFrame, skill_cols: list[str], control_dummies: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    y = df["log_salary"]

    # Model 1: Skills only (uncontrolled)
    model_skills = sm.OLS(y, sm.add_constant(df[skill_cols])).fit(cov_type="HC3")

    # Model 2: Skills + Controls (main specification)
    X_full = sm.add_constant(pd.concat([df[skill_cols], control_dummies], axis=1))
    model_full = sm.OLS(y, X_full).fit(cov_type="HC3")

    conf_int = model_full.conf_int(alpha=config["alpha"])
    rows = []
    for col in skill_cols:
        skill_key = col.removeprefix("skill_")

        display_names = {
            "cpp": "C++",
            "csharp": "C#",
            "ci_cd": "CI/CD",
            "llm_genai": "LLM / GenAI",
            "power_bi": "Power BI",
            "pytorch": "PyTorch",
            "tensorflow": "TensorFlow",
            "scikit_learn": "Scikit-Learn",
        }

        raw_label = skill_key.replace("_", " ")

        clean_name = display_names.get(
            skill_key,
            raw_label.upper() if len(raw_label) <= 4 else raw_label.title()
        )

        has_skill = df[col] == 1
        n_with = int(has_skill.sum())
        demand_pct = (n_with / len(df)) * 100

        # Simple (unadjusted) salary comparison
        med_with = df.loc[has_skill, "annual_salary"].median()
        med_without = df.loc[~has_skill, "annual_salary"].median()
        unadj_diff_pct = ((med_with - med_without) / med_without) * 100

        # Controlled regression association: (exp(beta) - 1) * 100
        beta = model_full.params[col]
        se = model_full.bse[col]
        pval = model_full.pvalues[col]
        ci_low_log, ci_high_log = conf_int.loc[col, 0], conf_int.loc[col, 1]

        adj_pct = (np.exp(beta) - 1) * 100
        ci_low_pct = (np.exp(ci_low_log) - 1) * 100
        ci_high_pct = (np.exp(ci_high_log) - 1) * 100

        rows.append(
            {
                "skill": clean_name,
                "skill_col": col,
                "job_count": n_with,
                "demand_pct": round(demand_pct, 2),
                "median_salary_with": round(med_with, 2),
                "median_salary_without": round(med_without, 2),
                "unadjusted_diff_pct": round(unadj_diff_pct, 2),
                "beta_log": round(beta, 5),
                "std_error_hc3": round(se, 5),
                "adjusted_assoc_pct": round(adj_pct, 2),
                "ci_95_lower_pct": round(ci_low_pct, 2),
                "ci_95_upper_pct": round(ci_high_pct, 2),
                "p_value": round(pval, 5),
                "significant_5pct": bool(pval < config["alpha"]),
            }
        )

    coef_df = pd.DataFrame(rows).sort_values("adjusted_assoc_pct", ascending=False).reset_index(drop=True)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "Model 1: Skills Only (Uncontrolled)",
                "n_obs": int(model_skills.nobs),
                "n_predictors": int(model_skills.df_model),
                "r_squared": round(model_skills.rsquared, 4),
                "adj_r_squared": round(model_skills.rsquared_adj, 4),
                "f_statistic": round(float(model_skills.fvalue), 2),
                "f_pvalue": float(model_skills.f_pvalue),
                "aic": round(model_skills.aic, 1),
                "bic": round(model_skills.bic, 1),
            },
            {
                "model": "Model 2: Skills + Controls (Main)",
                "n_obs": int(model_full.nobs),
                "n_predictors": int(model_full.df_model),
                "r_squared": round(model_full.rsquared, 4),
                "adj_r_squared": round(model_full.rsquared_adj, 4),
                "f_statistic": round(float(model_full.fvalue), 2),
                "f_pvalue": float(model_full.f_pvalue),
                "aic": round(model_full.aic, 1),
                "bic": round(model_full.bic, 1),
            },
        ]
    )

    print(
        f"[4/6] Regression complete -> Skills-only R² = {model_skills.rsquared:.4f} | "
        f"Controlled R² = {model_full.rsquared:.4f} (Adj R² = {model_full.rsquared_adj:.4f})"
    )
    return coef_df, metrics_df, model_full.summary().as_text()


# ==============================================================================
# 6. VISUALIZATIONS (3 REQUIRED PHASE 3 PLOTS)
# ==============================================================================
def generate_figures(df: pd.DataFrame, coef_df: pd.DataFrame, fig_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Salary distributions (raw vs log-transformed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(df["annual_salary"] / 1000, bins=35, color="#3b82f6", edgecolor="white", alpha=0.9)
    axes[0].axvline(
        df["annual_salary"].median() / 1000,
        color="#ef4444",
        linestyle="--",
        label=f"Median: ${df['annual_salary'].median():,.0f}",
    )
    axes[0].set_title("Raw Annual Salary Distribution (Right-Skewed)", fontweight="bold")
    axes[0].set_xlabel("Annual Salary ($ Thousands USD)")
    axes[0].set_ylabel("Job Postings")
    axes[0].legend()

    axes[1].hist(df["log_salary"], bins=35, color="#10b981", edgecolor="white", alpha=0.9)
    axes[1].axvline(
        df["log_salary"].mean(),
        color="#ef4444",
        linestyle="--",
        label=f"Mean log: {df['log_salary'].mean():.2f}",
    )
    axes[1].set_title("Log-Transformed Salary Distribution (Near-Normal)", fontweight="bold")
    axes[1].set_xlabel("log(Annual Salary)")
    axes[1].set_ylabel("Job Postings")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "salary_distributions.png", dpi=300)
    plt.close(fig)

    # 2. Regression coefficient plot (with 95% confidence intervals)
    plot_df = coef_df.sort_values("adjusted_assoc_pct", ascending=True)
    colors = ["#059669" if sig else "#94a3b8" for sig in plot_df["significant_5pct"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.28)))
    ax.axvline(0, color="#1e293b", linestyle="--", linewidth=1)
    ax.errorbar(
        plot_df["adjusted_assoc_pct"],
        plot_df["skill"],
        xerr=[
            plot_df["adjusted_assoc_pct"] - plot_df["ci_95_lower_pct"],
            plot_df["ci_95_upper_pct"] - plot_df["adjusted_assoc_pct"],
        ],
        fmt="none",
        ecolor="#64748b",
        elinewidth=1.2,
        capsize=2.5,
    )
    ax.scatter(plot_df["adjusted_assoc_pct"], plot_df["skill"], c=colors, s=45, zorder=3)
    ax.set_title("Controlled Salary Association by Skill (% Change, 95% HC3 CI)", fontweight="bold")
    ax.set_xlabel("Adjusted Salary Association (%) — Green: p < 0.05 | Gray: p ≥ 0.05")
    plt.tight_layout()
    fig.savefig(fig_dir / "regression_coefficients.png", dpi=300)
    plt.close(fig)

    # 3. Skill demand vs. salary association scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.axvline(coef_df["demand_pct"].median(), color="#64748b", linestyle=":", linewidth=1)

    sig_mask = coef_df["significant_5pct"]
    ax.scatter(
        coef_df.loc[~sig_mask, "demand_pct"],
        coef_df.loc[~sig_mask, "adjusted_assoc_pct"],
        color="#94a3b8",
        alpha=0.75,
        s=60,
        label="Not Significant (p ≥ 0.05)",
    )
    ax.scatter(
        coef_df.loc[sig_mask, "demand_pct"],
        coef_df.loc[sig_mask, "adjusted_assoc_pct"],
        color="#2563eb",
        alpha=0.9,
        s=80,
        label="Significant (p < 0.05)",
    )

    for _, row in coef_df.iterrows():
        ax.annotate(
            row["skill"],
            (row["demand_pct"], row["adjusted_assoc_pct"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Skill Market Demand vs. Controlled Salary Association", fontweight="bold")
    ax.set_xlabel("Market Demand (% of Job Postings Requiring Skill)")
    ax.set_ylabel("Adjusted Salary Association (%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "demand_vs_salary_scatter.png", dpi=300)
    plt.close(fig)
    print(f"[5/6] Saved 3 figures to {fig_dir}/")


# ==============================================================================
# 7. MAIN ENTRY POINT
# ==============================================================================
def main():
    CONFIG["regression_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figures_dir"].mkdir(parents=True, exist_ok=True)

    merged_df, all_skill_cols = load_merged_dataset(CONFIG)
    clean_df = clean_salary_sample(merged_df, CONFIG)
    feat_df, kept_skills, control_dummies = prepare_features(clean_df, all_skill_cols, CONFIG)
    feat_df.to_csv(
        CONFIG["regression_dir"] / "salary_analysis_sample.csv",
        index=False,
    )
    coef_df, metrics_df, summary_txt = run_regression_analysis(
        feat_df, kept_skills, control_dummies, CONFIG
    )

    coef_df.to_csv(CONFIG["regression_dir"] / "salary_coefficients.csv", index=False)
    metrics_df.to_csv(CONFIG["regression_dir"] / "salary_model_results.csv", index=False)
    (CONFIG["regression_dir"] / "regression_summary.txt").write_text(summary_txt, encoding="utf-8")

    generate_figures(feat_df, coef_df, CONFIG["figures_dir"])
    print(f"[6/6] All Phase 3 outputs exported successfully to {CONFIG['regression_dir']}/")


if __name__ == "__main__":
    main()
