from pathlib import Path
import re
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Skill Dynamics & Compensation in the Tech Job Market",
    page_icon="🛠️",
    layout="wide",
)


# Minimal styling for metric cards
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid color-mix(
            in srgb,
            var(--text-color) 15%,
            transparent
        );
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    div[data-testid="stMetric"] * {
        color: var(--text-color);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TEMPLATE = "plotly_dark"
COLORS = {
    "Community 1": "#38bdf8",
    "Community 2": "#fb923c",
    "Community 3": "#34d399",
}
DISPLAY_NAMES = {
    "python": "Python", "sql": "SQL", "r": "R", "go": "Go", "c++": "C++", "c#": "C#",
    "aws": "AWS", "gcp": "GCP", "ci/cd": "CI/CD", "llm / genai": "LLM / GenAI",
    "tensorflow": "TensorFlow", "power bi": "Power BI", "pytorch": "PyTorch",
    "numpy": "NumPy", "javascript": "JavaScript", "typescript": "TypeScript",
    "postgresql": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB",
    "bigquery": "BigQuery", "scikit-learn": "Scikit-Learn", "nlp": "NLP",
}


def clean_label(s: str) -> str:
    s = str(s).strip().lower()
    return DISPLAY_NAMES.get(s, s.title())


def to_col(s: str) -> str:
    s = {"c++": "cpp", "c#": "csharp"}.get(str(s).strip().lower(), str(s).strip().lower())
    return f"skill_{re.sub(r'[^a-z0-9]+', '_', s).strip('_')}"


@st.cache_data
def load_data():
    d2, d3, d4 = Path("Datasets"), Path("salary_model_results/regression"), Path("graph_analysis_results")

    combined = pd.read_csv(d2 / "final_combined_data_tech_only.csv")
    skills_clean = pd.read_csv(d2 / "job_skills_clean.csv")
    freq = pd.read_csv(d2 / "skill_frequency_stats.csv")
    by_source = pd.read_csv(d2 / "skill_by_source.csv")
    by_seniority = pd.read_csv(d2 / "skill_by_seniority.csv")
    by_workmode = pd.read_csv(d2 / "skill_by_workmode.csv")
    pairs = pd.read_csv(d2 / "skill_pair_stats.csv")

    sal_sample = pd.read_csv(d3 / "salary_analysis_sample.csv")
    sal_coef = pd.read_csv(d3 / "salary_coefficients.csv")
    sal_metrics = pd.read_csv(d3 / "salary_model_results.csv")

    centrality = pd.read_csv(d4 / "centrality_results.csv")
    edges = pd.read_csv(d4 / "network_edges.csv")
    community = pd.read_csv(d4 / "community_summary.csv")

    # Standardize skill IDs and display labels
    for df in [freq, by_source, by_seniority, by_workmode, centrality]:
        df["skill_id"] = df["skill"].astype(str).str.strip().str.lower()
        df["skill_label"] = df["skill_id"].map(clean_label)

    freq["skill_col"] = freq["skill_id"].map(to_col)

    for df in [pairs, edges]:
        df["skill_1"] = df["skill_1"].astype(str).str.strip().str.lower()
        df["skill_2"] = df["skill_2"].astype(str).str.strip().str.lower()
        df["label_1"] = df["skill_1"].map(clean_label)
        df["label_2"] = df["skill_2"].map(clean_label)

    # Master skill table merging Phase 2, 3, and 4
    cent_cols = ["skill_id", "degree", "weighted_degree", "pagerank", "betweenness", "community_id", "community_label"]
    master = (
        freq.merge(sal_coef.drop(columns=["skill"], errors="ignore"), on="skill_col", how="left", suffixes=("_full", "_sal"))
        .merge(centrality[[c for c in cent_cols if c in centrality.columns]], on="skill_id", how="left")
    )

    # 2D network coordinates from pre-computed weights
    G = nx.Graph()
    G.add_nodes_from(master["skill_id"])
    for r in edges.itertuples(index=False):
        G.add_edge(r.skill_1, r.skill_2, weight=float(r.job_count))
    pos = nx.spring_layout(G, seed=42, weight="weight", iterations=250)
    master["x"] = master["skill_id"].map(lambda s: float(pos[s][0]))
    master["y"] = master["skill_id"].map(lambda s: float(pos[s][1]))

    return combined, skills_clean, freq, by_source, by_seniority, by_workmode, sal_sample, sal_metrics, community, master, edges


combined_df, skills_clean_df, freq_df, by_source_df, by_seniority_df, by_workmode_df, sal_df, metrics_df, comm_df, master, edges = load_data()

st.title("🛠️ Skill Dynamics & Compensation in the Tech Job Market")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Full Market Overview",
    "2. Salary Boost Simulator",
    "3. Custom Skill & Role Filter",
    "4. Compensation Drivers & Pairs",
    "5. Skill Network & Communities",
])

# ==============================================================================
# TAB 1: FULL MARKET OVERVIEW
# ==============================================================================
with tab1:
    top_k = st.slider("Top Skills to Show", 10, 46, 20, width = 400)

    ov_jobs = combined_df 
    n_matched = ov_jobs["job_uid"].isin(set(skills_clean_df["job_uid"])).sum()

    # create five columns for each metric
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Postings", f"{len(ov_jobs):,}", f"{100*n_matched/len(ov_jobs):.1f}% with validated skills")
    m2.metric("Remote", f"{100*(ov_jobs['work_mode']=='Remote').mean():.1f}%", f"{(ov_jobs['work_mode']=='Remote').sum():,} jobs")
    m3.metric("Onsite", f"{100*(ov_jobs['work_mode']=='Onsite').mean():.1f}%", f"{(ov_jobs['work_mode']=='Onsite').sum():,} jobs")
    m4.metric("Hybrid", f"{100*(ov_jobs['work_mode']=='Hybrid').mean():.1f}%", f"{(ov_jobs['work_mode']=='Hybrid').sum():,} jobs")
    m5.metric("Clean USD Sample", f"{len(sal_df):,}", f"Median ${sal_df['annual_salary'].median():,.0f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Work Mode Split")
        wm = ov_jobs["work_mode"].value_counts().reset_index()
        fig = px.pie(wm, names="work_mode", values="count", hole=0.5, template=TEMPLATE)
        fig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Seniority Breakdown")
        sen = ov_jobs["level"].value_counts().reindex(["Intern", "Entry", "Mid", "Senior", "Lead+"]).dropna().reset_index()
        fig = px.bar(sen, x="level", y="count", text="count", color="level", template=TEMPLATE)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.subheader("Standardized Role Categories")
        rc = sal_df["role_category"].value_counts().reset_index()
        fig = px.bar(rc.iloc[::-1], x="count", y="role_category", orientation="h", text="count", template=TEMPLATE)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=25))
        st.plotly_chart(fig, use_container_width=True)

    r1, r2 = st.columns([1.15, 1])
    with r1:
        st.subheader(f"Top {top_k} Demanded Skills")
        sk_df = freq_df
        sk_df = sk_df.sort_values("demand_pct", ascending=False).head(top_k).iloc[::-1]
        fig = px.bar(sk_df, x="demand_pct", y="skill_label", orientation="h", text="job_count", template=TEMPLATE)
        fig.update_traces(texttemplate="%{x:.1f}% (%{text:,})", textposition="outside")
        fig.update_layout(height=500, xaxis_title="Share of Postings (%)", yaxis_title="", margin=dict(t=10, b=10, l=10, r=40))
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.subheader("Demand Heatmap by Segment")
        seg_pick = st.radio("Segment By", ["Seniority", "Work Mode", "Source"], horizontal=True)
        top12 = freq_df.sort_values("demand_pct", ascending=False).head(12)["skill_label"].tolist()

        if seg_pick == "Seniority":
            piv = by_seniority_df[by_seniority_df["level"] != "Intern"].pivot(index="skill_label", columns="level", values="demand_pct")
            piv = piv[[c for c in ["Entry", "Mid", "Senior", "Lead+"] if c in piv.columns]]
        elif seg_pick == "Work Mode":
            piv = by_workmode_df.pivot(index="skill_label", columns="work_mode", values="demand_pct")
        else:
            piv = by_source_df.pivot(index="skill_label", columns="source", values="demand_pct")

        fig = px.imshow(piv.reindex(top12), text_auto=".1f", color_continuous_scale="Blues", aspect="auto", template=TEMPLATE)
        fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: SALARY BOOST SIMULATOR ("WHAT SHOULD I LEARN NEXT?")
# ==============================================================================
with tab2:
    s1, s2, s3, s4 = st.columns([1.2, 1, 1, 1.8])
    sim_role = s1.selectbox("Target Role", sorted(sal_df["role_category"].unique()))
    sim_level = s2.selectbox("Seniority", ["Entry", "Mid", "Senior", "Lead+"], index=2)
    sim_mode = s3.selectbox("Work Mode", sorted(sal_df["work_mode"].unique()))
    known_skills = s4.multiselect("Skills You Already Know", sorted(master["skill_label"].unique()), default=["Python", "SQL"])

    # Baseline peer salary
    peers = sal_df[(sal_df["role_category"] == sim_role) & (sal_df["level"] == sim_level)]
    if peers.empty:
        peers = sal_df[sal_df["role_category"] == sim_role]
    base_salary = float(peers["annual_salary"].median())

    # Score unlearned skills
    label_to_id = dict(zip(master["skill_label"], master["skill_id"]))
    known_ids = {label_to_id[s] for s in known_skills if s in label_to_id}
    cands = master[~master["skill_label"].isin(known_skills)].dropna(subset=["adjusted_assoc_pct"]).copy()

    def avg_jaccard(cid):
        if not known_ids:
            return 0.0
        sub_e = edges[((edges["skill_1"] == cid) & (edges["skill_2"].isin(known_ids))) | ((edges["skill_2"] == cid) & (edges["skill_1"].isin(known_ids)))]
        return float(sub_e["jaccard"].mean()) if not sub_e.empty else 0.0

    cands["stack_fit"] = [avg_jaccard(cid) for cid in cands["skill_id"]]
    cands["dollar_boost"] = base_salary * (cands["adjusted_assoc_pct"] / 100.0)
    cands["projected_salary"] = base_salary + cands["dollar_boost"]

    if st.checkbox("Show only statistically significant pay drivers (p < 0.05)", value=True):
        cands = cands[cands["significant_5pct"]]
    cands = cands.sort_values("adjusted_assoc_pct", ascending=False)

    b1, b2, b3 = st.columns(3)
    b1.metric("Current Peer Baseline", f"${base_salary:,.0f}", f"{len(peers)} matching peer jobs")
    if not cands.empty:
        top_pay = cands.iloc[0]
        top_fit = cands.sort_values(["stack_fit", "adjusted_assoc_pct"], ascending=False).iloc[0]
        b2.metric("Highest Salary Booster", top_pay["skill_label"], f"+${top_pay['dollar_boost']:,.0f}/yr (+{top_pay['adjusted_assoc_pct']:.1f}%)")
        b3.metric("Best Natural Stack Fit", top_fit["skill_label"], f"Jaccard {top_fit['stack_fit']:.2f} | +${top_fit['dollar_boost']:,.0f}/yr")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Estimated Annual Salary Boost ($)")
        plot_c = cands.head(15).iloc[::-1]
        fig = px.bar(plot_c, x="dollar_boost", y="skill_label", orientation="h", text="adjusted_assoc_pct", template=TEMPLATE)
        fig.update_traces(texttemplate="+$%{x:,.0f} (%{text:+.1f}%)", textposition="outside", marker_color="#10b981")
        fig.update_layout(height=450, xaxis_title="Estimated Gain (USD)", yaxis_title="", margin=dict(t=10, b=10, l=10, r=65))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.subheader("Pay Lift vs. Compatibility with Your Stack")
        fig = px.scatter(
            cands, x="stack_fit", y="adjusted_assoc_pct", size="demand_pct_full",
            color="community_label", text="skill_label", color_discrete_map=COLORS, template=TEMPLATE,
            labels={"stack_fit": "Avg Jaccard with Your Skills", "adjusted_assoc_pct": "Controlled Salary Lift (%)"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=450, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 3: CUSTOM CONDITION & MULTI-SKILL FILTER
# ==============================================================================
with tab3:
    f1, f2, f3, f4 = st.columns(4)
    f_roles = f1.multiselect("Role", sorted(sal_df["role_category"].unique()), default=sorted(sal_df["role_category"].unique()))
    f_levels = f2.multiselect("Level", ["Entry", "Mid", "Senior", "Lead+"], default=["Entry", "Mid", "Senior", "Lead+"])
    f_modes = f3.multiselect("Work Mode", sorted(sal_df["work_mode"].unique()), default=sorted(sal_df["work_mode"].unique()))
    f_sources = f4.multiselect("Source", sorted(sal_df["source"].unique()), default=sorted(sal_df["source"].unique()))

    sk1, sk2 = st.columns([3, 1])
    req_skills = sk1.multiselect("Required Technical Skills", sorted(master["skill_label"].unique()))
    logic = sk2.radio("Match Mode", ["ALL (AND)", "ANY (OR)"], horizontal=True)

    sub = sal_df[
        sal_df["role_category"].isin(f_roles)
        & sal_df["level"].isin(f_levels)
        & sal_df["work_mode"].isin(f_modes)
        & sal_df["source"].isin(f_sources)
    ]
    if req_skills:
        cols = [to_col(s) for s in req_skills if to_col(s) in sub.columns]
        sub = sub[sub[cols].sum(axis=1) == len(cols)] if "ALL" in logic else sub[sub[cols].sum(axis=1) >= 1]

    if sub.empty:
        st.warning("No matching jobs found. Try broadening your filters.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Matching Postings", f"{len(sub):,}")
        k2.metric("Median Salary", f"${sub['annual_salary'].median():,.0f}")
        k3.metric("IQR (25th–75th)", f"${sub['annual_salary'].quantile(0.25):,.0f} – ${sub['annual_salary'].quantile(0.75):,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Salary Distribution")
            fig = px.histogram(sub, x="annual_salary", color="level", nbins=30, marginal="box", template=TEMPLATE)
            fig.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Top Companion Skills in Filtered Jobs")
            comp = [
                {"Skill": r.skill_label, "Jobs": int(sub[r.skill_col].sum()), "Share (%)": round(100 * sub[r.skill_col].mean(), 1)}
                for r in master.itertuples(index=False)
                if r.skill_label not in req_skills and r.skill_col in sub.columns and sub[r.skill_col].sum() > 0
            ]
            comp_df = pd.DataFrame(comp).sort_values("Jobs", ascending=False).head(15).iloc[::-1]
            fig = px.bar(comp_df, x="Share (%)", y="Skill", orientation="h", text="Jobs", template=TEMPLATE)
            fig.update_traces(texttemplate="%{x:.1f}% (%{text})", textposition="outside")
            fig.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=40))
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 4: REGRESSION DRIVERS & PAIR SYNERGIES
# ==============================================================================
with tab4:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Controlled Salary Association (95% HC3 CI)")
        reg = master.dropna(subset=["adjusted_assoc_pct"]).sort_values("adjusted_assoc_pct")
        fig = go.Figure()
        fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
        fig.add_trace(go.Scatter(
            x=reg["adjusted_assoc_pct"], y=reg["skill_label"], mode="markers",
            marker=dict(color=["#10b981" if s else "#64748b" for s in reg["significant_5pct"]], size=8),
            error_x=dict(type="data", symmetric=False, array=reg["ci_95_upper_pct"] - reg["adjusted_assoc_pct"], arrayminus=reg["adjusted_assoc_pct"] - reg["ci_95_lower_pct"], color="#64748b"),
        ))
        fig.update_layout(template=TEMPLATE, height=620, xaxis_title="Controlled Salary Lift (%)", margin=dict(t=10, b=10, l=10, r=15))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Demand vs. Salary Lift Quadrant")
        quad = master.dropna(subset=["adjusted_assoc_pct"])
        fig = px.scatter(
            quad, x="demand_pct_full", y="adjusted_assoc_pct", size="pagerank",
            color="community_label", text="skill_label", color_discrete_map=COLORS, template=TEMPLATE,
            labels={"demand_pct_full": "Market Demand (%)", "adjusted_assoc_pct": "Salary Lift (%)"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=310, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Highest-Paying Co-Occurring Pairs")
        pair_rows = []
        for r in edges.sort_values("job_count", ascending=False).head(25).itertuples(index=False):
            c1_c, c2_c = to_col(r.skill_1), to_col(r.skill_2)
            if c1_c in sal_df.columns and c2_c in sal_df.columns:
                m = (sal_df[c1_c] == 1) & (sal_df[c2_c] == 1)
                if m.sum() >= 10:
                    pair_rows.append({"Pair": f"{r.label_1} + {r.label_2}", "Median Salary": sal_df.loc[m, "annual_salary"].median(), "n": int(m.sum())})
        ps = pd.DataFrame(pair_rows).sort_values("Median Salary").tail(12)
        fig = px.bar(ps, x="Median Salary", y="Pair", orientation="h", text="n", template=TEMPLATE)
        fig.update_traces(texttemplate="$%{x:,.0f} (n=%{text})", textposition="outside")
        fig.update_layout(height=290, margin=dict(t=10, b=10, l=10, r=60))
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 5: SKILL NETWORK & COMMUNITIES
# ==============================================================================
with tab5:
    n1, n2 = st.columns([1.4, 1])
    with n1:
        g1, g2, g3 = st.columns(3)
        min_w = g1.slider("Min Co-Occurrence Weight", 20, 400, 111, 5)
        color_by = g2.selectbox("Node Color", ["Louvain Community", "Salary Lift (%)"])
        focus = g3.selectbox("Spotlight Skill", ["All Skills"] + sorted(master["skill_label"].unique()))

        vis_e = edges[edges["job_count"] >= min_w]
        if focus != "All Skills":
            vis_e = vis_e[(vis_e["label_1"] == focus) | (vis_e["label_2"] == focus)]

        pos_map = dict(zip(master["skill_label"], zip(master["x"], master["y"])))
        ex, ey = [], []
        for r in vis_e.itertuples(index=False):
            if r.label_1 in pos_map and r.label_2 in pos_map:
                ex.extend([pos_map[r.label_1][0], pos_map[r.label_2][0], None])
                ey.extend([pos_map[r.label_1][1], pos_map[r.label_2][1], None])

        fig_g = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="rgba(148,163,184,0.3)"), hoverinfo="none", showlegend=False))

        if color_by == "Louvain Community":
            for clab, grp in master.groupby("community_label"):
                fig_g.add_trace(go.Scatter(
                    x=grp["x"], y=grp["y"], mode="markers+text", name=clab, text=grp["skill_label"], textposition="top center",
                    marker=dict(size=10 + grp["demand_pct_full"] * 0.6, color=COLORS.get(clab, "#38bdf8")),
                ))
        else:
            fig_g.add_trace(go.Scatter(
                x=master["x"], y=master["y"], mode="markers+text", text=master["skill_label"], textposition="top center",
                marker=dict(size=10 + master["demand_pct_full"] * 0.6, color=master["adjusted_assoc_pct"].fillna(0), colorscale="RdYlGn", cmid=0, colorbar=dict(title="Lift %")),
            ))

        fig_g.update_layout(template=TEMPLATE, height=550, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_g, use_container_width=True)

    with n2:
        st.subheader("Centrality Leaderboard")
        top_c = master.sort_values("weighted_degree", ascending=False).head(12).iloc[::-1]
        fig_c = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=("Weighted Deg", "PageRank", "Betweenness"))
        b_cols = [COLORS.get(c, "#38bdf8") for c in top_c["community_label"]]
        fig_c.add_trace(go.Bar(x=top_c["weighted_degree"], y=top_c["skill_label"], orientation="h", marker_color=b_cols), 1, 1)
        fig_c.add_trace(go.Bar(x=top_c["pagerank"], y=top_c["skill_label"], orientation="h", marker_color=b_cols), 1, 2)
        fig_c.add_trace(go.Bar(x=top_c["betweenness"], y=top_c["skill_label"], orientation="h", marker_color=b_cols), 1, 3)
        fig_c.update_layout(template=TEMPLATE, height=340, showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_c, use_container_width=True)

        st.subheader("Louvain Community Summary")
        st.dataframe(comm_df.drop(columns=["all_skills"], errors="ignore"), use_container_width=True, hide_index=True)