from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

CONFIG = {
    "skills_file": Path("Datasets/job_skills_clean.csv"),
    "jobs_file": Path("Datasets/final_combined_data_tech_only.csv"),
    "regression_dir": Path("salary_model_results/regression"),
    "figures_dir": Path("salary_model_results/figures"),
    "target_currency": "USD",
    "min_salary_usd": 15_000,
    "max_salary_usd": 600_000,
    "exclude_levels": ["Intern", "Internship"],
    "min_skill_count": 20,
    "control_cols": ["level", "work_mode", "source", "role_category"],
    "alpha": 0.05,
}


def load_merged_dataset(config: dict) -> tuple[pd.DataFrame, list[str]]:
    skills_long = pd.read_csv(config["skills_file"])
    jobs_df = pd.read_csv(config["jobs_file"])
    print(f"[1/5] Loaded {len(jobs_df):,} jobs and {len(skills_long):,} skill rows")

    skill_labels = (
        skills_long["skill"].fillna("").astype(str).str.strip().str.lower()
        .replace({"c++": "cpp", "c#": "csharp"})
    )
    skills_long["skill_clean"] = skill_labels.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    skills_long = skills_long[skills_long["skill_clean"] != ""].drop_duplicates(["job_uid", "skill_clean"])

    skill_matrix = (
        pd.crosstab(skills_long["job_uid"], skills_long["skill_clean"])
        .clip(upper=1)
        .add_prefix("skill_")
        .reset_index()
    )

    merged = jobs_df.merge(skill_matrix, on="job_uid", how="left")
    all_skill_cols = [c for c in merged.columns if c.startswith("skill_")]
    merged[all_skill_cols] = merged[all_skill_cols].fillna(0).astype(int)
    return merged, all_skill_cols


def clean_salary_sample(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = df.copy()
    initial_n = len(data)

    # 1. Currency & non-missing salary filter
    if "currency" in data.columns and config["target_currency"]:
        data = data[data["currency"].astype(str).str.upper() == config["target_currency"]].copy()
    data["annual_salary"] = pd.to_numeric(data["annual_salary"], errors="coerce")
    data = data.dropna(subset=["annual_salary"]).copy()

    # 2. Exclude internships
    if "level" in data.columns and config["exclude_levels"]:
        data = data[~data["level"].astype(str).isin(config["exclude_levels"])].copy()

    # 3. Hard USD salary range check ($15,000 - $600,000)
    data = data[
        (data["annual_salary"] >= config["min_salary_usd"])
        & (data["annual_salary"] <= config["max_salary_usd"])
    ].copy()

    # 4. Log transformation
    data["log_salary"] = np.log(data["annual_salary"])

    print(
        f"[2/5] Salary sample: {initial_n:,} -> Clean USD n={len(data):,} | "
        f"Median=${data['annual_salary'].median():,.0f} | Mean=${data['annual_salary'].mean():,.0f}"
    )
    return data


def derive_role_category(df: pd.DataFrame) -> pd.Series:
    # Combine parsed categories tags + title into one clean lowercase string
    cats = (
        df.get("categories", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[\[\]'\"\-_/]+", " ", regex=True)
    )
    titles = (
        df.get("title", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[-_/]+", " ", regex=True)
    )
    combined = (cats + " " + titles).str.replace(r"\s+", " ", regex=True).str.strip()

    conditions = [
        # 1. AI / ML / Data Science
        combined.str.contains(
            r"\bmachine learning\b|\bml engineer|\bai engineer|\bai ml\b|\bgenerative ai\b|\bllm\b|\bdeep learning\b|\bcomputer vision\b|\bnlp\b|\bdata scien|\bapplied scientist\b|\bresearch scientist\b",
            regex=True,
        ),
        # 2. Data / Analytics Engineering
        combined.str.contains(
            r"\bdata engineer|\bbig data engineer|\banalytics engineer|\bdata architect\b|\bdata platform\b|\bdata infrastructure\b|\bdatabase\b|\betl\b|\bdbt\b",
            regex=True,
        ),
        # 3. BI / Data & Product Analytics
        combined.str.contains(
            r"\bbi\b|\bbusiness intelligence\b|\btableau\b|\bpower bi\b|\bdata analyst\b|\bbusiness analyst\b|\bproduct analyst\b|\breporting analyst\b|\bdata and analytics\b",
            regex=True,
        ),
        # 4. Cloud / DevOps / Platform / Security
        combined.str.contains(
            r"\bdevops\b|\bsre\b|\bsite reliability\b|\bcloud engineer|\bplatform engineer|\binfrastructure\b|\bkubernetes\b|\bsecurity\b|\bcybersecurity\b",
            regex=True,
        ),
        # 5. Software Engineering (Backend / Full-Stack / Frontend / Mobile)
        combined.str.contains(
            r"\bsoftware engineer|\bsoftware engineering\b|\bbackend\b|\bback end\b|\bfrontend\b|\bfront end\b|\bfull stack\b|\bfullstack\b|\bdeveloper\b|\bweb engineer\b",
            regex=True,
        ),
    ]

    choices = [
        "Data Science / ML / AI",
        "Data / Analytics Eng",
        "Data Analyst / BI",
        "Cloud / DevOps / Security",
        "Software Engineering",
    ]

    return pd.Series(np.select(conditions, choices, default="Other Technical / IT"), index=df.index)


def prepare_features(
    df: pd.DataFrame, all_skill_cols: list[str], config: dict
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    data = df.copy()
    kept_skills = [c for c in all_skill_cols if data[c].sum() >= config["min_skill_count"]]
    print(f"[3/5] Kept {len(kept_skills)} skills (>= {config['min_skill_count']} postings)")

    data["role_category"] = derive_role_category(data)
    print(data["role_category"].value_counts())

    active_controls = []
    for col in config["control_cols"]:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
            if data[col].nunique() > 1:
                active_controls.append(col)

    control_dummies = pd.get_dummies(data[active_controls], drop_first=True, dtype=int)
    print(f"      Controls: {active_controls} -> {control_dummies.shape[1]} dummy columns")
    return data, kept_skills, control_dummies


def run_regression_analysis(
    df: pd.DataFrame, skill_cols: list[str], control_dummies: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    y = df["log_salary"]
    model_skills = sm.OLS(y, sm.add_constant(df[skill_cols])).fit(cov_type="HC3")
    X_full = sm.add_constant(pd.concat([df[skill_cols], control_dummies], axis=1))
    model_full = sm.OLS(y, X_full).fit(cov_type="HC3")

    conf_int = model_full.conf_int(alpha=config["alpha"])
    display_names = {
        "cpp": "C++", "csharp": "C#", "ci_cd": "CI/CD", "llm_genai": "LLM / GenAI",
        "power_bi": "Power BI", "pytorch": "PyTorch", "tensorflow": "TensorFlow", "scikit_learn": "Scikit-Learn",
    }

    rows = []
    for col in skill_cols:
        skill_key = col.removeprefix("skill_")
        raw_label = skill_key.replace("_", " ")
        clean_name = display_names.get(skill_key, raw_label.upper() if len(raw_label) <= 4 else raw_label.title())

        has_skill = df[col] == 1
        n_with = int(has_skill.sum())
        med_with = df.loc[has_skill, "annual_salary"].median()
        med_without = df.loc[~has_skill, "annual_salary"].median()

        beta = model_full.params[col]
        pval = model_full.pvalues[col]
        ci_low_log, ci_high_log = conf_int.loc[col, 0], conf_int.loc[col, 1]

        rows.append({
            "skill": clean_name,
            "skill_col": col,
            "job_count": n_with,
            "demand_pct": round((n_with / len(df)) * 100, 2),
            "median_salary_with": round(med_with, 2),
            "median_salary_without": round(med_without, 2),
            "unadjusted_diff_pct": round(((med_with - med_without) / med_without) * 100, 2),
            "beta_log": round(beta, 5),
            "std_error_hc3": round(model_full.bse[col], 5),
            "adjusted_assoc_pct": round((np.exp(beta) - 1) * 100, 2),
            "ci_95_lower_pct": round((np.exp(ci_low_log) - 1) * 100, 2),
            "ci_95_upper_pct": round((np.exp(ci_high_log) - 1) * 100, 2),
            "p_value": round(pval, 5),
            "significant_5pct": bool(pval < config["alpha"]),
        })

    coef_df = pd.DataFrame(rows).sort_values("adjusted_assoc_pct", ascending=False).reset_index(drop=True)
    metrics_df = pd.DataFrame([
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
    ])

    print(f"[4/5] Regression complete -> Controlled R² = {model_full.rsquared:.4f} (Adj R² = {model_full.rsquared_adj:.4f})")
    return coef_df, metrics_df, model_full.summary().as_text()


def generate_figures(df: pd.DataFrame, coef_df: pd.DataFrame, fig_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Salary distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(df["annual_salary"] / 1000, bins=35, color="#3b82f6", edgecolor="white", alpha=0.9)
    axes[0].axvline(df["annual_salary"].median() / 1000, color="#ef4444", linestyle="--", label=f"Median: ${df['annual_salary'].median():,.0f}")
    axes[0].set_title("Raw Annual Salary Distribution", fontweight="bold")
    axes[0].set_xlabel("Annual Salary ($ Thousands USD)")
    axes[0].legend()

    axes[1].hist(df["log_salary"], bins=35, color="#10b981", edgecolor="white", alpha=0.9)
    axes[1].axvline(df["log_salary"].mean(), color="#ef4444", linestyle="--", label=f"Mean log: {df['log_salary'].mean():.2f}")
    axes[1].set_title("Log-Transformed Salary Distribution", fontweight="bold")
    axes[1].set_xlabel("log(Annual Salary)")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "salary_distributions.png", dpi=300)
    plt.close(fig)

    # 2. Regression coefficient forest plot
    plot_df = coef_df.sort_values("adjusted_assoc_pct", ascending=True)
    colors = ["#059669" if sig else "#94a3b8" for sig in plot_df["significant_5pct"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.28)))
    ax.axvline(0, color="#1e293b", linestyle="--", linewidth=1)
    ax.errorbar(
        plot_df["adjusted_assoc_pct"], plot_df["skill"],
        xerr=[plot_df["adjusted_assoc_pct"] - plot_df["ci_95_lower_pct"], plot_df["ci_95_upper_pct"] - plot_df["adjusted_assoc_pct"]],
        fmt="none", ecolor="#64748b", elinewidth=1.2, capsize=2.5,
    )
    ax.scatter(plot_df["adjusted_assoc_pct"], plot_df["skill"], c=colors, s=45, zorder=3)
    ax.set_title("Controlled Salary Association by Skill (% Change, 95% HC3 CI)", fontweight="bold")
    ax.set_xlabel("Adjusted Salary Association (%) — Green: p < 0.05 | Gray: p ≥ 0.05")
    plt.tight_layout()
    fig.savefig(fig_dir / "regression_coefficients.png", dpi=300)
    plt.close(fig)

    # 3. Demand vs. salary scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.axvline(coef_df["demand_pct"].median(), color="#64748b", linestyle=":", linewidth=1)
    sig_mask = coef_df["significant_5pct"]
    ax.scatter(coef_df.loc[~sig_mask, "demand_pct"], coef_df.loc[~sig_mask, "adjusted_assoc_pct"], color="#94a3b8", alpha=0.75, s=60, label="p ≥ 0.05")
    ax.scatter(coef_df.loc[sig_mask, "demand_pct"], coef_df.loc[sig_mask, "adjusted_assoc_pct"], color="#2563eb", alpha=0.9, s=80, label="p < 0.05")
    for _, row in coef_df.iterrows():
        ax.annotate(row["skill"], (row["demand_pct"], row["adjusted_assoc_pct"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_title("Skill Market Demand vs. Controlled Salary Association", fontweight="bold")
    ax.set_xlabel("Market Demand (% of Postings)")
    ax.set_ylabel("Adjusted Salary Association (%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "demand_vs_salary_scatter.png", dpi=300)
    plt.close(fig)


def main():
    CONFIG["regression_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figures_dir"].mkdir(parents=True, exist_ok=True)

    merged_df, all_skill_cols = load_merged_dataset(CONFIG)
    clean_df = clean_salary_sample(merged_df, CONFIG)
    feat_df, kept_skills, control_dummies = prepare_features(clean_df, all_skill_cols, CONFIG)

    feat_df.to_csv(CONFIG["regression_dir"] / "salary_analysis_sample.csv", index=False)
    coef_df, metrics_df, summary_txt = run_regression_analysis(feat_df, kept_skills, control_dummies, CONFIG)

    coef_df.to_csv(CONFIG["regression_dir"] / "salary_coefficients.csv", index=False)
    metrics_df.to_csv(CONFIG["regression_dir"] / "salary_model_results.csv", index=False)
    (CONFIG["regression_dir"] / "regression_summary.txt").write_text(summary_txt, encoding="utf-8")

    generate_figures(feat_df, coef_df, CONFIG["figures_dir"])
    print(f"[5/5] Exported all Phase 3 outputs to {CONFIG['regression_dir']}/")


if __name__ == "__main__":
    main()