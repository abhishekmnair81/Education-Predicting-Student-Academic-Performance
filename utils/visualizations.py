import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Uniform Luxury Palette
COLOR_PASS = "#10B981"      # Emerald green
COLOR_FAIL = "#EF4444"      # Vibrant crimson
COLOR_PRIMARY = "#3B82F6"   # Electric blue
COLOR_SECONDARY = "#6366F1" # Indigo
COLOR_ACCENT = "#8B5CF6"    # Purple
COLOR_WARNING = "#F59E0B"   # Amber
COLOR_MUTED = "#94A3B8"     # Slate muted
COLOR_TEXT = "#0F172A"      # Slate 900
COLOR_GRID = "#F1F5F9"      # Very soft slate grid

PALETTE_STATUS = {"Pass": COLOR_PASS, "Fail": COLOR_FAIL}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#0F172A", size=12),
    margin=dict(l=40, r=30, t=55, b=45),
    hoverlabel=dict(
        bgcolor="#0F172A",
        bordercolor="#334155",
        font_size=12,
        font_family="Inter, sans-serif",
        font_color="#FFFFFF"
    )
)

def create_pass_fail_donut(df):
    """
    1. Pass vs Fail count & donut chart with center metric and dynamic insight.
    """
    if len(df) == 0:
        return go.Figure(), "No data available under current filter criteria."

    counts = df["Final_Status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]

    color_map = [PALETTE_STATUS.get(s, COLOR_PRIMARY) for s in counts["Status"]]
    pull_map = [0.03 if s == "Fail" else 0.0 for s in counts["Status"]]

    pass_count = int((df["Target"] == 1).sum())
    fail_count = int((df["Target"] == 0).sum())
    pass_pct = (pass_count / len(df)) * 100 if len(df) > 0 else 0.0

    fig = go.Figure(data=[go.Pie(
        labels=counts["Status"],
        values=counts["Count"],
        hole=0.64,
        pull=pull_map,
        marker=dict(colors=color_map, line=dict(color="#FFFFFF", width=2.5)),
        textinfo="label+percent",
        textfont=dict(family="Inter", size=12, color="#FFFFFF"),
        hovertemplate="<b>%{label} Cohort</b><br>Students: %{value}<br>Share: %{percent}<extra></extra>"
    )])

    # Center callout inside the donut hole
    fig.add_annotation(
        text=f"<b style='font-size:24px;color:#0F172A;'>{pass_pct:.1f}%</b><br><span style='font-size:11px;color:#64748B;font-weight:600;'>PASS RATE</span>",
        x=0.5, y=0.5,
        font=dict(family="Inter, sans-serif"),
        showarrow=False
    )

    fig.update_layout(
        title=dict(text="<b>Pass vs Fail Outcome Distribution</b>", font=dict(size=15, color="#0F172A")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS
    )

    insight = (
        f"💡 <b>Cohort Health:</b> Among {len(df)} filtered students, <b>{pass_pct:.1f}%</b> ({pass_count}) "
        f"are on track to pass, while <b>{100 - pass_pct:.1f}%</b> ({fail_count}) require proactive early warning intervention."
    )
    return fig, insight

def create_g3_histogram(df):
    """
    2. G3 final grade distribution histogram with vertical pass mark at 10.
    """
    if len(df) == 0:
        return go.Figure(), "No data available."

    fig = go.Figure()
    
    # Fail distribution
    fail_df = df[df["Target"] == 0]
    pass_df = df[df["Target"] == 1]

    fig.add_trace(go.Histogram(
        x=fail_df["G3"],
        name="Fail (G3 < 10)",
        marker=dict(color=COLOR_FAIL, line=dict(color="#FFFFFF", width=1)),
        opacity=0.9,
        xbins=dict(start=0, end=21, size=1),
        hovertemplate="Grade %{x}: <b>%{y}</b> failing students<extra></extra>"
    ))

    fig.add_trace(go.Histogram(
        x=pass_df["G3"],
        name="Pass (G3 ≥ 10)",
        marker=dict(color=COLOR_PASS, line=dict(color="#FFFFFF", width=1)),
        opacity=0.9,
        xbins=dict(start=0, end=21, size=1),
        hovertemplate="Grade %{x}: <b>%{y}</b> passing students<extra></extra>"
    ))

    # Pass threshold line at 9.5
    fig.add_vline(
        x=9.5,
        line_width=2.5,
        line_dash="dash",
        line_color="#0F172A",
        annotation_text="Pass Cutoff (10.0 pts)",
        annotation_position="top left",
        annotation_font=dict(color="#0F172A", size=11, family="Inter", weight="bold"),
        annotation_bgcolor="rgba(255, 255, 255, 0.9)",
        annotation_borderpad=3
    )

    avg_g3 = df["G3"].mean()
    med_g3 = df["G3"].median()

    fig.update_layout(
        barmode="stack",
        title=dict(text="<b>Final Exam Grade Distribution (G3: 0–20 Scale)</b>", font=dict(size=15, color="#0F172A")),
        xaxis=dict(title="Final Grade (G3)", dtick=2, gridcolor=COLOR_GRID, zeroline=False),
        yaxis=dict(title="Student Count", gridcolor=COLOR_GRID, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS
    )

    failing_zeroes = int((df["G3"] == 0).sum())
    insight = (
        f"💡 <b>Grade Trajectory:</b> Average final grade is <b>{avg_g3:.2f}/20</b> (Median: {med_g3:.1f}). "
        f"Notably, <b>{failing_zeroes} students</b> received an absolute 0 grade, indicating course abandonment or complete disengagement."
    )
    return fig, insight

def create_failures_vs_outcome_chart(df):
    """
    3. Past failures vs outcome (grouped bar).
    """
    if len(df) == 0:
        return go.Figure(), "No data available."

    grouped = df.groupby(["failures", "Final_Status"], observed=False).size().unstack(fill_value=0).reset_index()
    for col in ["Pass", "Fail"]:
        if col not in grouped.columns:
            grouped[col] = 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["failures"],
        y=grouped["Pass"],
        name="Pass",
        marker=dict(color=COLOR_PASS, line=dict(color="#FFFFFF", width=1)),
        hovertemplate="Failures: %{x}<br>Pass Count: <b>%{y}</b><extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=grouped["failures"],
        y=grouped["Fail"],
        name="Fail",
        marker=dict(color=COLOR_FAIL, line=dict(color="#FFFFFF", width=1)),
        hovertemplate="Failures: %{x}<br>Fail Count: <b>%{y}</b><extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Academic History: Past Course Failures vs Outcome</b>", font=dict(size=15, color="#0F172A")),
        xaxis=dict(title="Number of Past Course Failures (0–3)", dtick=1, gridcolor=COLOR_GRID),
        yaxis=dict(title="Student Count", gridcolor=COLOR_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS
    )

    zero_fail_pass_rate = (
        (df[df["failures"] == 0]["Target"] == 1).mean() * 100 if (df["failures"] == 0).any() else 0.0
    )
    prior_fail_pass_rate = (
        (df[df["failures"] > 0]["Target"] == 1).mean() * 100 if (df["failures"] > 0).any() else 0.0
    )

    insight = (
        f"💡 <b>Failure Persistence:</b> Students with 0 prior failures enjoy a <b>{zero_fail_pass_rate:.1f}%</b> pass rate, "
        f"which plummets to <b>{prior_fail_pass_rate:.1f}%</b> for students with 1 or more past course failures."
    )
    return fig, insight

def create_absences_boxplot(df):
    """
    4. Absences by outcome (box plot).
    """
    if len(df) == 0:
        return go.Figure(), "No data available."

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df[df["Final_Status"] == "Pass"]["absences"],
        name="Pass Cohort",
        marker_color=COLOR_PASS,
        line=dict(color=COLOR_PASS, width=2),
        fillcolor="rgba(16, 185, 129, 0.2)",
        boxpoints="outliers",
        jitter=0.3,
        pointpos=-1.8
    ))
    fig.add_trace(go.Box(
        y=df[df["Final_Status"] == "Fail"]["absences"],
        name="Fail / At-Risk Cohort",
        marker_color=COLOR_FAIL,
        line=dict(color=COLOR_FAIL, width=2),
        fillcolor="rgba(239, 68, 68, 0.2)",
        boxpoints="outliers",
        jitter=0.3,
        pointpos=-1.8
    ))

    fig.update_layout(
        title=dict(text="<b>Attendance Proxy: Absences Distribution by Outcome</b>", font=dict(size=15, color="#0F172A")),
        yaxis=dict(title="School Absences (Days)", gridcolor=COLOR_GRID),
        xaxis=dict(gridcolor=COLOR_GRID),
        showlegend=False,
        **LAYOUT_DEFAULTS
    )

    max_abs = int(df["absences"].max()) if len(df) > 0 else 0

    insight = (
        f"💡 <b>Attendance Signal:</b> Failing students exhibit high variance in absenteeism (up to <b>{max_abs} days</b>). "
        f"While chronic absenteeism (>10 days) strongly elevates failure probability, zero-absence clusters also require monitoring for dropouts."
    )
    return fig, insight

def create_studytime_pass_rate_chart(df):
    """
    5. Study time vs pass rate (bar).
    """
    if len(df) == 0:
        return go.Figure(), "No data available."

    mapping = {1: "< 2 hrs/wk", 2: "2 to 5 hrs/wk", 3: "5 to 10 hrs/wk", 4: "> 10 hrs/wk"}
    df_st = df.copy()
    df_st["study_label"] = df_st["studytime"].map(mapping).fillna("Unknown")

    grouped = df_st.groupby("studytime", observed=False).agg(
        pass_rate=("Target", "mean"),
        total=("Target", "count"),
        study_label=("study_label", "first")
    ).reset_index()
    grouped["pass_pct"] = grouped["pass_rate"] * 100

    fig = go.Figure(data=[go.Bar(
        x=grouped["study_label"],
        y=grouped["pass_pct"],
        text=[f"<b>{p:.1f}%</b><br><span style='font-size:10px;'>({n} students)</span>" for p, n in zip(grouped["pass_pct"], grouped["total"])],
        textposition="outside",
        marker=dict(
            color=grouped["pass_pct"],
            colorscale=[[0, COLOR_FAIL], [0.5, COLOR_WARNING], [1, COLOR_PASS]],
            cmin=40,
            cmax=90,
            line=dict(color="#FFFFFF", width=1.5)
        ),
        hovertemplate="Study Category: <b>%{x}</b><br>Pass Rate: <b>%{y:.1f}%</b><extra></extra>"
    )])

    fig.update_layout(
        title=dict(text="<b>Pass Rate by Weekly Study Commitment</b>", font=dict(size=15, color="#0F172A")),
        xaxis=dict(title="Weekly Study Hours Category", gridcolor=COLOR_GRID),
        yaxis=dict(title="Pass Rate (%)", range=[0, 110], dtick=20, gridcolor=COLOR_GRID),
        **LAYOUT_DEFAULTS
    )

    lowest_rate = grouped.loc[grouped["studytime"] == 1, "pass_pct"].values[0] if (grouped["studytime"] == 1).any() else 0
    highest_rate = grouped.loc[grouped["studytime"] >= 3, "pass_pct"].mean() if (grouped["studytime"] >= 3).any() else 0

    insight = (
        f"💡 <b>Study Hours Leverage:</b> Increasing study time from <2 hrs (<b>{lowest_rate:.1f}%</b> pass rate) "
        f"to >5 hrs yields an average pass rate of <b>{highest_rate:.1f}%</b> (+{highest_rate - lowest_rate:.1f}% gain)."
    )
    return fig, insight

def create_correlation_heatmap(df):
    """
    6. Correlation heatmap of key numerical and target attributes.
    """
    corr_cols = [
        "failures", "studytime", "Medu", "Fedu", "Dalc", "Walc",
        "absences", "age", "traveltime", "famrel", "G1", "G2", "Target"
    ]
    available_cols = [c for c in corr_cols if c in df.columns]

    if len(df) < 2 or len(available_cols) < 2:
        return go.Figure(), "Insufficient data for correlation analysis."

    corr_df = df[available_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-0.8,
        zmax=0.8,
        text=np.round(corr_df.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, family="Inter", color="#0F172A"),
        hoverongaps=False,
        hovertemplate="Feature X: <b>%{x}</b><br>Feature Y: <b>%{y}</b><br>Correlation r: <b>%{z:.3f}</b><extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Correlation Matrix: Academic, Habitual, and Social Drivers</b>", font=dict(size=15, color="#0F172A")),
        xaxis=dict(tickangle=-45, gridcolor=COLOR_GRID),
        yaxis=dict(gridcolor=COLOR_GRID),
        **LAYOUT_DEFAULTS
    )

    if "Target" in corr_df.columns:
        target_corrs = corr_df["Target"].drop("Target")
        top_pos = target_corrs.idxmax()
        top_pos_val = target_corrs.max()
        top_neg = target_corrs.idxmin()
        top_neg_val = target_corrs.min()
        insight = (
            f"💡 <b>Correlation Dynamics:</b> Strongest positive driver of passing is <b>{top_pos}</b> (r = +{top_pos_val:.2f}), "
            f"while the strongest negative risk driver is <b>{top_neg}</b> (r = {top_neg_val:.2f})."
        )
    else:
        insight = "💡 Correlation calculated across numerical attributes."

    return fig, insight

def create_categorical_pass_rates_chart(df):
    """
    7. Pass rate by sex, school, address, higher ambition, internet, family support.
    """
    if len(df) == 0:
        return go.Figure(), "No data available."

    factors = [
        ("School", "school", {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}),
        ("Sex", "sex", {"F": "Female", "M": "Male"}),
        ("Address", "address", {"U": "Urban", "R": "Rural"}),
        ("Higher Ed Ambition", "higher", {"yes": "Aims Higher Ed", "no": "No Ambition"}),
        ("Internet Access", "internet", {"yes": "Home Internet", "no": "No Internet"}),
        ("Family Support", "famsup", {"yes": "Family Support", "no": "No Family Support"})
    ]

    records = []
    for factor_title, col, label_dict in factors:
        if col in df.columns:
            agg = df.groupby(col, observed=False)["Target"].agg(["mean", "count"]).reset_index()
            for _, row in agg.iterrows():
                val = row[col]
                readable_label = label_dict.get(val, str(val))
                records.append({
                    "Factor": factor_title,
                    "Group": f"{factor_title}: {readable_label}",
                    "Pass_Rate": row["mean"] * 100,
                    "Students": row["count"]
                })

    fact_df = pd.DataFrame(records)
    if len(fact_df) == 0:
        return go.Figure(), "No categorical demographic attributes found."

    fig = go.Figure(data=[go.Bar(
        x=fact_df["Pass_Rate"],
        y=fact_df["Group"],
        orientation="h",
        text=[f"<b>{p:.1f}%</b> ({n})" for p, n in zip(fact_df["Pass_Rate"], fact_df["Students"])],
        textposition="outside",
        marker=dict(
            color=fact_df["Pass_Rate"],
            colorscale=[[0, COLOR_FAIL], [0.65, COLOR_WARNING], [1, COLOR_PASS]],
            cmin=40,
            cmax=80,
            line=dict(color="#FFFFFF", width=1.5)
        ),
        hovertemplate="<b>%{y}</b><br>Pass Rate: <b>%{x:.1f}%</b><extra></extra>"
    )])

    fig.update_layout(
        title=dict(text="<b>Pass Rate Equity Gaps Across Demographic & Institutional Factors</b>", font=dict(size=15, color="#0F172A")),
        xaxis=dict(title="Pass Rate (%)", range=[0, 108], dtick=20, gridcolor=COLOR_GRID),
        yaxis=dict(autorange="reversed", gridcolor=COLOR_GRID),
        **LAYOUT_DEFAULTS
    )

    higher_pass = df[df["higher"] == "yes"]["Target"].mean() * 100 if "higher" in df.columns else 0
    no_higher_pass = df[df["higher"] == "no"]["Target"].mean() * 100 if "higher" in df.columns else 0

    insight = (
        f"💡 <b>Equity Divide:</b> Aspirations play a massive role: students aspiring to higher education pass at "
        f"<b>{higher_pass:.1f}%</b>, compared to only <b>{no_higher_pass:.1f}%</b> for those without ambition (gap of {higher_pass - no_higher_pass:.1f}%)."
    )
    return fig, insight

def create_g1_g2_scatter(df):
    """
    8. Scatter: G1 vs G2 colored by outcome with risk quadrants.
    """
    if len(df) == 0 or "G1" not in df.columns or "G2" not in df.columns:
        return go.Figure(), "G1 and G2 grades unavailable."

    fig = go.Figure()

    for status, color in [("Pass", COLOR_PASS), ("Fail", COLOR_FAIL)]:
        sub = df[df["Final_Status"] == status]
        fig.add_trace(go.Scatter(
            x=sub["G1"],
            y=sub["G2"],
            mode="markers",
            name=f"Outcome: {status}",
            marker=dict(
                size=9,
                color=color,
                opacity=0.85,
                line=dict(width=1.5, color="#FFFFFF")
            ),
            text=[
                f"Student ID #{idx}<br>G1: {g1}<br>G2: {g2}<br>G3: {g3}<br>Absences: {ab}"
                for idx, g1, g2, g3, ab in zip(sub.index, sub["G1"], sub["G2"], sub["G3"], sub["absences"])
            ],
            hovertemplate="<b>%{text}</b><extra></extra>"
        ))

    # Reference quadrants at Grade 10
    fig.add_vline(x=9.5, line_width=1.5, line_dash="dot", line_color=COLOR_MUTED)
    fig.add_hline(y=9.5, line_width=1.5, line_dash="dot", line_color=COLOR_MUTED)

    fig.update_layout(
        title=dict(text="<b>Early Assessment Correlation: Period 1 (G1) vs Period 2 (G2)</b>", font=dict(size=15, color="#0F172A")),
        xaxis=dict(title="Period 1 Grade (G1)", dtick=2, gridcolor=COLOR_GRID),
        yaxis=dict(title="Period 2 Grade (G2)", dtick=2, gridcolor=COLOR_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS
    )

    r_val = df[["G1", "G2"]].corr().iloc[0, 1]
    g2_fail_eventual_fail = (
        (df[df["G2"] < 10]["Target"] == 0).mean() * 100 if (df["G2"] < 10).any() else 0.0
    )

    insight = (
        f"💡 <b>Grade Momentum:</b> G1 and G2 are exceptionally correlated (r = {r_val:.2f}). "
        f"Crucially, <b>{g2_fail_eventual_fail:.1f}%</b> of students scoring under 10 in Period 2 ultimately fail the final examination."
    )
    return fig, insight

def create_confusion_matrix_charts(cm_dt, cm_svm):
    """
    Side-by-side interactive confusion matrices for Decision Tree and SVM.
    """
    labels = ["Fail (At-Risk)", "Pass"]

    def plot_single_cm(cm, title, colorscale, primary_hex):
        total = cm.sum()
        cm_pct = (cm / total) * 100
        text = [
            [f"<b style='font-size:16px;'>{val}</b><br><span style='font-size:11px;'>({pct:.1f}%)</span>" for val, pct in zip(row_val, row_pct)]
            for row_val, row_pct in zip(cm, cm_pct)
        ]
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=colorscale,
            text=text,
            texttemplate="%{text}",
            textfont=dict(family="Inter", size=13),
            showscale=False,
            hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: <b>%{z}</b><extra></extra>"
        ))
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#0F172A")),
            xaxis=dict(title="Predicted Academic Class", gridcolor=COLOR_GRID),
            yaxis=dict(title="Actual Ground Truth", autorange="reversed", gridcolor=COLOR_GRID),
            **LAYOUT_DEFAULTS
        )
        return fig

    fig_dt = plot_single_cm(cm_dt, "Decision Tree Confusion Matrix", "Blues", COLOR_PRIMARY)
    fig_svm = plot_single_cm(cm_svm, "SVM (RBF Kernel) Confusion Matrix", "Purples", COLOR_SECONDARY)
    return fig_dt, fig_svm

def create_roc_curve_chart(roc_dt, roc_svm):
    """
    ROC Curves comparison with AUC score in legend.
    """
    fpr_dt, tpr_dt, auc_dt = roc_dt
    fpr_svm, tpr_svm, auc_svm = roc_svm

    fig = go.Figure()

    # Fill area for lead model (DT)
    fig.add_trace(go.Scatter(
        x=fpr_dt,
        y=tpr_dt,
        mode="lines",
        name=f"Decision Tree (AUC = {auc_dt:.4f})",
        line=dict(color=COLOR_PRIMARY, width=3.5),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.08)",
        hovertemplate="Decision Tree<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=fpr_svm,
        y=tpr_svm,
        mode="lines",
        name=f"SVM - RBF Kernel (AUC = {auc_svm:.4f})",
        line=dict(color=COLOR_PASS, width=3),
        hovertemplate="SVM<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
    ))

    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Guess (AUC = 0.5000)",
        line=dict(color=COLOR_MUTED, width=1.5, dash="dash"),
        hoverinfo="skip"
    ))

    fig.update_layout(
        title=dict(text="<b>Receiver Operating Characteristic (ROC) Benchmark</b>", font=dict(size=15, color="#0F172A")),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02], gridcolor=COLOR_GRID),
        yaxis=dict(title="True Positive Rate (Recall / Sensitivity)", range=[-0.02, 1.02], gridcolor=COLOR_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS
    )
    return fig

def create_metrics_comparison_bar(metrics_dt, metrics_svm):
    """
    Grouped bar chart comparing metrics across both models.
    """
    metric_keys = ["Accuracy", "Precision (Pass)", "Recall (Pass)", "Recall (Fail - Risk)", "F1-Score", "ROC-AUC", "5-Fold CV Accuracy"]
    vals_dt = [metrics_dt[k] * 100 for k in metric_keys]
    vals_svm = [metrics_svm[k] * 100 for k in metric_keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metric_keys,
        y=vals_dt,
        name="Decision Tree",
        marker=dict(color=COLOR_PRIMARY, line=dict(color="#FFFFFF", width=1)),
        text=[f"<b>{v:.1f}%</b>" for v in vals_dt],
        textposition="outside",
        hovertemplate="Decision Tree - %{x}: <b>%{y:.2f}%</b><extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=metric_keys,
        y=vals_svm,
        name="SVM (RBF Kernel)",
        marker=dict(color=COLOR_SECONDARY, line=dict(color="#FFFFFF", width=1)),
        text=[f"<b>{v:.1f}%</b>" for v in vals_svm],
        textposition="outside",
        hovertemplate="SVM - %{x}: <b>%{y:.2f}%</b><extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Comparative Performance Profile: Decision Tree vs SVM</b>", font=dict(size=15, color="#0F172A")),
        yaxis=dict(title="Metric Score (%)", range=[0, 115], dtick=20, gridcolor=COLOR_GRID),
        xaxis=dict(tickangle=-20, gridcolor=COLOR_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS
    )
    return fig

def create_feature_importance_bar(feat_imp, top_n=10):
    """
    Top N feature importances horizontal bar chart.
    """
    top_feats = feat_imp.head(top_n).iloc[::-1]

    fig = go.Figure(data=[go.Bar(
        x=top_feats.values,
        y=top_feats.index,
        orientation="h",
        text=[f"<b>{v:.3f}</b>" for v in top_feats.values],
        textposition="outside",
        marker=dict(
            color=top_feats.values,
            colorscale=[[0, "#93C5FD"], [0.5, COLOR_PRIMARY], [1, COLOR_SECONDARY]],
            showscale=False,
            line=dict(color="#FFFFFF", width=1)
        ),
        hovertemplate="Feature: <b>%{y}</b><br>Gini Importance: <b>%{x:.4f}</b><extra></extra>"
    )])

    fig.update_layout(
        title=dict(text=f"<b>Top {top_n} Decision Tree Feature Importances (Gini Index)</b>", font=dict(size=15, color="#0F172A")),
        xaxis=dict(title="Relative Gini Importance Score", gridcolor=COLOR_GRID),
        yaxis=dict(title="Predictor Variable", gridcolor=COLOR_GRID),
        **LAYOUT_DEFAULTS
    )
    return fig

def create_risk_gauge(risk_score):
    """
    Speedometer / gauge indicator for What-If Student Risk Simulator.
    risk_score: 0.0 to 1.0 (Fail Probability).
    """
    pct = risk_score * 100
    if pct >= 70:
        bar_color = COLOR_FAIL
        band_text = "CRITICAL RISK (Fail Prob ≥ 70%)"
    elif pct >= 40:
        bar_color = COLOR_WARNING
        band_text = "MODERATE RISK (Fail Prob 40–70%)"
    else:
        bar_color = COLOR_PASS
        band_text = "LOW RISK (Pass Prob ≥ 60%)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(size=38, color="#0F172A", family="Inter, sans-serif")),
        title=dict(text=f"<b>Academic Failure Risk Index</b><br><span style='font-size:12px;color:#64748B;'>{band_text}</span>", font=dict(size=14, color="#0F172A")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94A3B8"),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor="#F1F5F9",
            borderwidth=1,
            bordercolor="#CBD5E1",
            steps=[
                dict(range=[0, 40], color="rgba(16, 185, 129, 0.18)"),
                dict(range=[40, 70], color="rgba(245, 158, 11, 0.18)"),
                dict(range=[70, 100], color="rgba(239, 68, 68, 0.18)")
            ],
            threshold=dict(
                line=dict(color="#0F172A", width=3),
                thickness=0.8,
                value=pct
            )
        )
    ))

    fig.update_layout(
        height=270,
        margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    return fig
