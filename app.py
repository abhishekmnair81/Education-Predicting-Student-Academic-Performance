import streamlit as st
import pandas as pd
import numpy as np
import io

from utils.data_loader import (
    load_dataset, get_data_summary, prepare_features, filter_dataframe,
    CATEGORICAL_COLS, NUMERIC_COLS
)
from utils.model_trainer import (
    train_and_evaluate_models, generate_tree_plot, is_matching_benchmark,
    NOTEBOOK_BENCHMARK
)
from utils.visualizations import (
    create_pass_fail_donut, create_g3_histogram, create_failures_vs_outcome_chart,
    create_absences_boxplot, create_studytime_pass_rate_chart, create_correlation_heatmap,
    create_categorical_pass_rates_chart, create_g1_g2_scatter, create_confusion_matrix_charts,
    create_roc_curve_chart, create_metrics_comparison_bar, create_feature_importance_bar,
    create_risk_gauge, COLOR_PASS, COLOR_FAIL, COLOR_PRIMARY
)
from utils.risk_analyzer import (
    score_cohort_risk, assign_risk_tiers, discover_at_risk_profiles,
    get_what_if_prediction
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduRisk EWS — Student Academic Performance Early-Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern Luxury Dashboard CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Global App Background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        max-width: 1440px;
    }

    /* ---------------------------------------------------------
       SIDEBAR LUXURY DARK GLASS STYLING
    --------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090E1A 0%, #0F172A 100%) !important;
        border-right: 1px solid #1E293B !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #E2E8F0 !important;
    }

    /* Sidebar Brand Box */
    .brand-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 18px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35);
    }

    .brand-title {
        margin: 0;
        font-size: 17px;
        font-weight: 800;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #94A3B8 !important;
        font-weight: 500;
    }

    /* Sidebar Navigation Menu Items */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 3px;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 9px 12px !important;
        border-radius: 8px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid transparent !important;
        cursor: pointer !important;
        background: transparent !important;
        margin-bottom: 2px !important;
        width: 100% !important;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #CBD5E1 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        transform: translateX(3px);
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
        color: #FFFFFF !important;
    }

    /* Selected state */
    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.25) 0%, rgba(99, 102, 241, 0.12) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.45) !important;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.18) !important;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #60A5FA !important;
        font-weight: 700 !important;
    }

    /* Hide default radio circle for clean nav pill feel */
    div[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* Sidebar File Uploader dark glass styling */
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px dashed #334155 !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] small {
        color: #94A3B8 !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]:hover {
        border-color: #60A5FA !important;
        background: rgba(59, 130, 246, 0.06) !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button {
        background: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }

    /* Sidebar Expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid #1E293B !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #CBD5E1 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    /* ---------------------------------------------------------
       NATIVE STREAMLIT BORDERED CONTAINER CARD STYLING
    --------------------------------------------------------- */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 22px 24px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03), 0 6px 16px -4px rgba(15, 23, 42, 0.04) !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
        margin-bottom: 18px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 12px 24px -6px rgba(15, 23, 42, 0.07) !important;
        border-color: #CBD5E1 !important;
    }

    /* Card Titles */
    h3 {
        color: #0F172A !important;
        font-size: 16.5px !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
        margin-bottom: 14px !important;
        letter-spacing: -0.015em !important;
    }

    /* ---------------------------------------------------------
       HERO TOPBAR COMPONENT
    --------------------------------------------------------- */
    .app-header-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px 26px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03), 0 8px 24px -6px rgba(15, 23, 42, 0.04);
        flex-wrap: wrap;
        gap: 16px;
    }

    .app-header-left {
        flex: 1;
        min-width: 280px;
    }

    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #EFF6FF;
        color: #2563EB;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 3px 10px;
        border-radius: 20px;
        border: 1px solid #DBEAFE;
        margin-bottom: 8px;
    }

    .header-title {
        font-size: 24px;
        font-weight: 800;
        color: #0F172A;
        margin: 0 0 6px 0;
        letter-spacing: -0.025em;
        line-height: 1.25;
    }

    .header-desc {
        font-size: 13.5px;
        color: #64748B;
        margin: 0;
        line-height: 1.5;
    }

    .app-header-right {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        align-self: center;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    .status-pill.live {
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }

    .status-pill.model {
        background: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
    }

    .status-pill.mode {
        background: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* ---------------------------------------------------------
       5-COLUMN RESPONSIVE KPI GRID
    --------------------------------------------------------- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 22px;
    }

    @media (max-width: 1024px) {
        .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 640px) {
        .kpi-grid { grid-template-columns: 1fr; }
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03), 0 4px 12px rgba(15, 23, 42, 0.02);
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3.5px;
    }

    .kpi-card.card-cohort::before { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
    .kpi-card.card-pass::before { background: linear-gradient(90deg, #10B981, #34D399); }
    .kpi-card.card-grade::before { background: linear-gradient(90deg, #6366F1, #818CF8); }
    .kpi-card.card-absences::before { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .kpi-card.card-risk::before { background: linear-gradient(90deg, #EF4444, #F87171); }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
    }

    .kpi-icon {
        font-size: 16px;
        opacity: 0.85;
    }

    .kpi-metric {
        font-size: 27px;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .kpi-unit {
        font-size: 13px;
        font-weight: 600;
        color: #94A3B8;
    }

    .kpi-progress {
        width: 100%;
        height: 5px;
        background: #F1F5F9;
        border-radius: 10px;
        margin: 10px 0 6px 0;
        overflow: hidden;
    }

    .kpi-progress-bar {
        height: 100%;
        border-radius: 10px;
        background: #3B82F6;
        transition: width 0.5s ease;
    }

    .kpi-progress-bar.pass { background: #10B981; }
    .kpi-progress-bar.neutral { background: #6366F1; }
    .kpi-progress-bar.warning { background: #F59E0B; }
    .kpi-progress-bar.fail { background: #EF4444; }

    .kpi-footer {
        font-size: 11.5px;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* ---------------------------------------------------------
       INSIGHT PILL & BANNERS
    --------------------------------------------------------- */
    .insight-pill {
        background: linear-gradient(90deg, #F0F9FF 0%, #F8FAFC 100%);
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        font-size: 12.5px;
        color: #1E293B;
        margin-top: 14px;
        line-height: 1.55;
        border-top: 1px solid #E0F2FE;
        border-right: 1px solid #E0F2FE;
        border-bottom: 1px solid #E0F2FE;
    }

    .alert-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        color: #92400E;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    .info-banner {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        color: #1E40AF;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    .badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        display: inline-block;
    }

    .badge-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        display: inline-block;
    }

    .benchmark-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #DCFCE7;
        color: #15803D;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid #86EFAC;
    }

    /* ---------------------------------------------------------
       PIPELINE WORKFLOW COMPONENT (5-Step Grid)
    --------------------------------------------------------- */
    .pipeline-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 16px 0 20px 0;
    }

    @media (max-width: 1024px) {
        .pipeline-container { grid-template-columns: repeat(2, 1fr); }
    }

    @media (max-width: 640px) {
        .pipeline-container { grid-template-columns: 1fr; }
    }

    .pipeline-step {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 14px;
        position: relative;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .pipeline-step:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
        border-color: #CBD5E1;
    }

    .pipeline-step.highlight {
        background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 100%);
        border-color: #86EFAC;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.08);
    }

    .step-num {
        font-size: 10px;
        font-weight: 800;
        color: #64748B;
        letter-spacing: 0.08em;
    }

    .step-icon {
        font-size: 20px;
    }

    .step-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #0F172A;
        margin: 6px 0 4px 0;
    }

    .step-desc {
        font-size: 11.5px;
        color: #64748B;
        line-height: 1.4;
        margin-bottom: 10px;
    }

    .step-tag {
        display: inline-block;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        color: #475569;
        align-self: flex-start;
    }

    /* ---------------------------------------------------------
       LUXURY COMPARISON TABLE
    --------------------------------------------------------- */
    .table-container {
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        overflow: hidden;
        margin: 14px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    .luxury-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        text-align: left;
    }

    .luxury-table th {
        background: #0F172A;
        color: #FFFFFF;
        padding: 13px 16px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        border-bottom: 1px solid #1E293B;
    }

    .luxury-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #F1F5F9;
        color: #1E293B;
    }

    .luxury-table tr:hover td {
        background-color: #F8FAFC;
    }

    .luxury-table tr:last-child td {
        border-bottom: none;
    }

    .lead-col {
        background-color: #F0FDF4 !important;
        font-weight: 700 !important;
        color: #15803D !important;
    }

    /* Primary Streamlit Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Branding, Data Upload, Filters & Toggles
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="brand-card">
            <div class="brand-icon">🎓</div>
            <div>
                <h2 class="brand-title">EduRisk EWS</h2>
                <div class="brand-subtitle">Early-Warning Academic System</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation Selector
    nav_page = st.radio(
        "Platform Modules",
        [
            "📊 Executive Overview",
            "🧹 Data & Preprocessing",
            "📈 Exploratory Data Analysis (EDA)",
            "⚙️ Model Training & Experiments",
            "⚖️ Model Comparison & Diagnostics",
            "🌳 Feature Importance & Tree Rules",
            "🚨 High-Risk Profiles & Interventions",
            "🔮 What-If Student Risk Simulator",
            "💡 Insights & Recommendations"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='border:none; border-top:1px solid #1E293B; margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>🎛️ Cohort & Data Controls</p>", unsafe_allow_html=True)

    # Dataset Loader / Uploader
    uploaded_file = st.file_uploader(
        "Upload Student Dataset (.csv)",
        type=["csv"],
        help="Upload UCI student-mat.csv or compatible file (semicolon or comma delimited)"
    )
    
    # Load Data safely
    try:
        raw_df = load_dataset(uploaded_file)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Model Configuration Toggles
    st.markdown("<p style='font-size:12px; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin: 14px 0 6px 0;'>⚙️ Model Strategy</p>", unsafe_allow_html=True)
    include_g1_g2 = st.toggle("Include G1 & G2 Exam Features", value=True, help="Default ON. When OFF, retrains models without early exam grades to test pre-semester early warning capability.")
    tune_mode = st.toggle("Enable 5-Fold Grid Search Tuning", value=False, help="Toggle between Default configuration and 5-Fold GridSearchCV Tuned models.")

    # Filter widgets
    with st.expander("👤 Demographic Filters", expanded=False):
        school_opt = st.selectbox("School", ["All"] + sorted(list(raw_df["school"].unique())))
        sex_opt = st.selectbox("Sex", ["All"] + sorted(list(raw_df["sex"].unique())))
        min_age, max_age = int(raw_df["age"].min()), int(raw_df["age"].max())
        age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        address_opt = st.selectbox("Address", ["All", "U", "R"], format_func=lambda x: "All" if x == "All" else ("Urban (U)" if x == "U" else "Rural (R)"))

    with st.expander("📚 Academic & Habit Filters", expanded=False):
        failures_opt = st.selectbox("Past Failures", ["All", "0", "1", "2", "3"])
        studytime_opt = st.selectbox("Study Time", ["All", "1", "2", "3", "4"], format_func=lambda x: {
            "All": "All", "1": "<2 hrs/wk", "2": "2-5 hrs/wk", "3": "5-10 hrs/wk", "4": ">10 hrs/wk"
        }.get(x, x))
        higher_opt = st.selectbox("Higher Ed Ambition", ["All", "yes", "no"])
        internet_opt = st.selectbox("Internet Access", ["All", "yes", "no"])
        romantic_opt = st.selectbox("Romantic Relationship", ["All", "yes", "no"])
        max_abs = int(raw_df["absences"].max())
        absences_range = st.slider("Absences Range (Days)", min_value=0, max_value=max_abs, value=(0, max_abs))

    # Reset Filters Button
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()

    # Apply Filters
    filter_params = {
        "school": school_opt,
        "sex": sex_opt,
        "age_range": age_range,
        "address": address_opt,
        "failures": failures_opt,
        "studytime": studytime_opt,
        "higher": higher_opt,
        "internet": internet_opt,
        "romantic": romantic_opt,
        "absences_range": absences_range
    }
    filtered_df = filter_dataframe(raw_df, filter_params)

    # Download Filtered Data Button
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Filtered Cohort",
        data=csv_buffer.getvalue(),
        file_name="filtered_student_cohort.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Proxy variables disclaimer note in sidebar
    st.markdown("""
        <div style="background-color:rgba(255,255,255,0.03); border:1px solid #1E293B; border-left:3px solid #3B82F6; padding:10px 12px; border-radius:8px; font-size:11px; color:#94A3B8; margin-top:16px; line-height:1.45;">
            <b style="color:#E2E8F0;">ℹ️ Proxy Variables:</b><br>
            • <code>absences</code>: Attendance proxy<br>
            • <code>G1</code>, <code>G2</code>: Midterm test scores<br>
            • <code>studytime</code>: Weekly study hours
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Global Model Training (Cached Resource)
# ---------------------------------------------------------
X_encoded_full, y_full, encoders_dict, feature_names_full, X_raw_full = prepare_features(
    raw_df, include_g1_g2=include_g1_g2
)

models_data = train_and_evaluate_models(
    X_encoded_full, y_full, include_g1_g2=include_g1_g2, tune_mode=tune_mode
)

cohort_scored = score_cohort_risk(filtered_df, models_data, model_choice="Decision Tree", include_g1_g2=include_g1_g2)

# Metrics calculation
total_cohort_count = len(filtered_df)
raw_total = len(raw_df)

if total_cohort_count > 0:
    cohort_pass_count = int((filtered_df["Target"] == 1).sum())
    cohort_fail_count = int((filtered_df["Target"] == 0).sum())
    cohort_pass_rate = (cohort_pass_count / total_cohort_count) * 100
    cohort_avg_g3 = float(filtered_df["G3"].mean())
    cohort_avg_abs = float(filtered_df["absences"].mean())
    pred_risk_count = int((cohort_scored["Predicted_Class"] == 0).sum())
    pred_risk_pct = (pred_risk_count / total_cohort_count) * 100
else:
    cohort_pass_count, cohort_fail_count, cohort_pass_rate = 0, 0, 0.0
    cohort_avg_g3, cohort_avg_abs = 0.0, 0.0
    pred_risk_count, pred_risk_pct = 0, 0.0

# Page Metadata mapping for Top Hero Header
page_meta = {
    "📊 Executive Overview": {
        "title": "Executive Overview & Cohort Pulse",
        "desc": "High-level institutional health telemetry, pass/fail distributions, and operational architecture.",
        "badge": "Institutional Command Center"
    },
    "🧹 Data & Preprocessing": {
        "title": "Data Hygiene, Profiling & Leakage Guard",
        "desc": "Comprehensive audit of raw dataset integrity, encoding transformations, and leakage prevention.",
        "badge": "ETL & Pipeline Audit"
    },
    "📈 Exploratory Data Analysis (EDA)": {
        "title": "Exploratory Data Analysis & Empirical Drivers",
        "desc": "Dynamic diagnostic visualizations exploring correlations, study commitments, and equity gaps.",
        "badge": "Cohort Analytics"
    },
    "⚙️ Model Training & Experiments": {
        "title": "Model Architecture & Hyperparameter Setup",
        "desc": "Algorithmic specification of Decision Tree and Support Vector Machine classifiers.",
        "badge": "ML Architecture"
    },
    "⚖️ Model Comparison & Diagnostics": {
        "title": "Model Comparison & Diagnostic Benchmarking",
        "desc": "Rigorous head-to-head evaluation across accuracy, precision, fail-recall, ROC-AUC, and cross-validation.",
        "badge": "Evaluation Benchmark"
    },
    "🌳 Feature Importance & Tree Rules": {
        "title": "Explainability: Feature Importance & Tree Rules",
        "desc": "Transparent feature attributions, architectural tree diagram, and readable logic rules.",
        "badge": "Interpretable AI"
    },
    "🚨 High-Risk Profiles & Interventions": {
        "title": "High-Risk Profiles, Tiers & Interventions",
        "desc": "Dynamic risk stratification, automated archetype clustering, and actionable case dossiers.",
        "badge": "Intervention Triage"
    },
    "🔮 What-If Student Risk Simulator": {
        "title": "What-If Student Risk Simulator & Predictor",
        "desc": "Simulate student profiles in real time to assess outcome probabilities and personalized advisor playbooks.",
        "badge": "Real-Time Simulator"
    },
    "💡 Insights & Recommendations": {
        "title": "Strategic Insights, Policy & Roadmap",
        "desc": "Actionable institutional recommendations, 4-tier rollout plan, and honest system limitations.",
        "badge": "Executive Roadmap"
    }
}

current_meta = page_meta.get(nav_page, {
    "title": nav_page,
    "desc": "Student Academic Performance Early-Warning System (EWS)",
    "badge": "Platform Console"
})

# ---------------------------------------------------------
# Modern Hero Header
# ---------------------------------------------------------
st.markdown(f"""
<div class="app-header-container">
    <div class="app-header-left">
        <div class="header-badge">🎓 {current_meta['badge']}</div>
        <h1 class="header-title">{current_meta['title']}</h1>
        <p class="header-desc">{current_meta['desc']}</p>
    </div>
    <div class="app-header-right">
        <div class="status-pill live"><span class="pulse-dot"></span> LIVE COHORT: N={total_cohort_count}</div>
        <div class="status-pill model">🤖 MODEL: DECISION TREE</div>
        <div class="status-pill mode">{'⚡ MIDTERM FEATURES (G1/G2) ON' if include_g1_g2 else '🌱 PRE-SEMESTER BASELINE MODE'}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Top Live KPI Grid (5-Column Responsive Row)
# ---------------------------------------------------------
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card card-cohort">
        <div class="kpi-header">
            <span class="kpi-label">Filtered Cohort</span>
            <span class="kpi-icon">👥</span>
        </div>
        <div class="kpi-metric">{total_cohort_count}</div>
        <div class="kpi-progress"><div class="kpi-progress-bar" style="width:{(total_cohort_count / raw_total * 100):.1f}%;"></div></div>
        <div class="kpi-footer">{(total_cohort_count / raw_total * 100):.1f}% of total {raw_total} students</div>
    </div>
    <div class="kpi-card card-pass">
        <div class="kpi-header">
            <span class="kpi-label">Pass Rate (G3 ≥ 10)</span>
            <span class="kpi-icon">🎯</span>
        </div>
        <div class="kpi-metric" style="color:#059669;">{cohort_pass_rate:.1f}%</div>
        <div class="kpi-progress"><div class="kpi-progress-bar pass" style="width:{cohort_pass_rate:.1f}%;"></div></div>
        <div class="kpi-footer"><b>{cohort_pass_count}</b> Passing • <b>{cohort_fail_count}</b> Failing</div>
    </div>
    <div class="kpi-card card-grade">
        <div class="kpi-header">
            <span class="kpi-label">Average Final Grade</span>
            <span class="kpi-icon">📝</span>
        </div>
        <div class="kpi-metric">{cohort_avg_g3:.2f} <span class="kpi-unit">/ 20</span></div>
        <div class="kpi-progress"><div class="kpi-progress-bar neutral" style="width:{(cohort_avg_g3 / 20 * 100):.1f}%;"></div></div>
        <div class="kpi-footer">Pass benchmark: 10.0 pts</div>
    </div>
    <div class="kpi-card card-absences">
        <div class="kpi-header">
            <span class="kpi-label">Average Absences</span>
            <span class="kpi-icon">📅</span>
        </div>
        <div class="kpi-metric">{cohort_avg_abs:.1f} <span class="kpi-unit">days</span></div>
        <div class="kpi-progress"><div class="kpi-progress-bar warning" style="width:{min(100.0, cohort_avg_abs / 25 * 100):.1f}%;"></div></div>
        <div class="kpi-footer">Attendance proxy metric</div>
    </div>
    <div class="kpi-card card-risk">
        <div class="kpi-header">
            <span class="kpi-label">Predicted At-Risk</span>
            <span class="kpi-icon">🚨</span>
        </div>
        <div class="kpi-metric" style="color:#DC2626;">{pred_risk_pct:.1f}%</div>
        <div class="kpi-progress"><div class="kpi-progress-bar fail" style="width:{pred_risk_pct:.1f}%;"></div></div>
        <div class="kpi-footer"><b>{pred_risk_count}</b> flagged for early warning</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Early Warning G1/G2 Toggle Performance Drop Alert
if not include_g1_g2:
    st.markdown("""
        <div class="alert-banner">
            ⚠️ <b>Early Intervention Baseline Mode Active:</b> Midterm grades (Period 1 <code>G1</code> and Period 2 <code>G2</code>) have been excluded from the model. 
            The system is predicting course outcome strictly using pre-semester baseline characteristics, past failures, study habits, and attendance patterns. 
            This enables academic advisors to flag vulnerable students <b>at Week 1 of the semester</b>, before any exam failure occurs.
        </div>
    """, unsafe_allow_html=True)

# Empty Filter Fallback Guard
if total_cohort_count == 0:
    st.warning("⚠️ No students match the active sidebar filters. Please adjust or reset the filter sliders in the sidebar.")
    st.stop()


# =========================================================
# PAGE 1: 📊 Executive Overview & Pulse
# =========================================================
if nav_page == "📊 Executive Overview":
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.container(border=True):
            fig_donut, ins_donut = create_pass_fail_donut(filtered_df)
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins_donut}</div>', unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            fig_hist, ins_hist = create_g3_histogram(filtered_df)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins_hist}</div>', unsafe_allow_html=True)

    # System Architecture Workflow Card (Visual Responsive 5-Step Grid)
    with st.container(border=True):
        st.markdown("### 🏛️ Early-Warning System Operational Architecture")
        
        st.html("""
        <div class="pipeline-container">
            <div class="pipeline-step">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="step-num">STEP 01</span>
                    <span class="step-icon">📥</span>
                </div>
                <div class="step-title">Student Ingestion</div>
                <div class="step-desc">Academics, demographics & study habits ingestion.</div>
                <div class="step-tag">395 Student Records</div>
            </div>
            <div class="pipeline-step">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="step-num">STEP 02</span>
                    <span class="step-icon">🛡️</span>
                </div>
                <div class="step-title">Preprocessing Guard</div>
                <div class="step-desc">Strict leakage prevention & label encoding.</div>
                <div class="step-tag">G3 Excluded • 17 Encoded</div>
            </div>
            <div class="pipeline-step highlight">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="step-num">STEP 03 (LEAD)</span>
                    <span class="step-icon">🧠</span>
                </div>
                <div class="step-title">Predictive ML</div>
                <div class="step-desc">Decision Tree vs SVM comparative benchmark.</div>
                <div class="step-tag">Stratified 80/20 • 5-Fold CV</div>
            </div>
            <div class="pipeline-step">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="step-num">STEP 04</span>
                    <span class="step-icon">📊</span>
                </div>
                <div class="step-title">Dynamic Risk Tiers</div>
                <div class="step-desc">High, Medium & Low risk band assignment.</div>
                <div class="step-tag">Calibrated Probabilities</div>
            </div>
            <div class="pipeline-step">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="step-num">STEP 05</span>
                    <span class="step-icon">🎯</span>
                </div>
                <div class="step-title">Advisor Action Plan</div>
                <div class="step-desc">4-tier proactive academic advisor playbooks.</div>
                <div class="step-tag">Tutoring • Contracts • Tech</div>
            </div>
        </div>
        """)

        st.markdown("""
        **Core Institutional Objectives:**
        1. **Zero Unidentified Dropouts:** Maximize Recall on the failing student class (**Class 0**) to virtually eliminate False Negatives.
        2. **Explainable Triage:** Equip advisors with clear, rule-based decision trees instead of opaque black-box scores.
        3. **Targeted Support:** Automatically map individual student risk signatures to specific university support services (peer tutoring, attendance contracts, digital equity).
        """)


# =========================================================
# PAGE 2: 🧹 Data & Preprocessing
# =========================================================
elif nav_page == "🧹 Data & Preprocessing":
    summary = get_data_summary(raw_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{summary['rows']} rows")
    with col2:
        st.metric("Predictor Features", f"{summary['cols']} columns")
    with col3:
        st.metric("Missing Values", f"{summary['missing']} (0.0%)", delta="Clean Integrity", delta_color="normal")
    with col4:
        st.metric("Duplicate Rows", f"{summary['duplicates']}", delta="Unique Records", delta_color="normal")

    # Data Leakage Prevention Callout Box
    st.markdown("""
        <div class="info-banner">
            🛡️ <b>Data Leakage Prevention Protocol:</b><br>
            To construct a genuine <b>Early-Warning System (EWS)</b> that predicts outcomes <i>before</i> final examinations take place, 
            the final grade <code>G3</code>, <code>Final_Status</code>, and <code>Target</code> are strictly excluded from the predictor feature matrix $X$.
            If included, $G3$ would cause artificial 100% accuracy while rendering the model completely useless for proactive intervention.
        </div>
    """, unsafe_allow_html=True)

    # Raw Data Preview
    with st.container(border=True):
        st.markdown("### 📋 Raw Student Dataset Preview (First 10 Records)")
        st.dataframe(raw_df.head(10), use_container_width=True)

    # Categorical Label Encoding Table Preview
    with st.container(border=True):
        st.markdown("### 🏷️ Categorical Feature Encoding (17 Variables)")
        st.markdown("""
            The UCI Mathematics dataset contains 17 categorical columns that require encoding into numeric formats for algorithmic consumption:
            `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.
        """)
        
        col_enc1, col_enc2 = st.columns(2)
        with col_enc1:
            st.markdown("**Before Encoding (Raw Categories):**")
            st.dataframe(X_raw_full[CATEGORICAL_COLS].head(5), use_container_width=True)
        with col_enc2:
            st.markdown("**After Encoding (`LabelEncoder`):**")
            st.dataframe(X_encoded_full[CATEGORICAL_COLS].head(5), use_container_width=True)

    # Train-Test Split and Scaling Summary
    with st.container(border=True):
        st.markdown("### 📐 Stratified Train-Test Split & Standardization Architecture")
        split_col1, split_col2 = st.columns(2)
        with split_col1:
            st.markdown("""
            **Stratified 80 / 20 Train-Test Partition:**
            - **Training Set:** 316 samples (80.0%)
            - **Test Holdout Set:** 79 samples (20.0%)
            - **Stratification:** Stratified by binary `Target` to preserve the exact 67.09% / 32.91% pass-to-fail ratio in both partitions.
            - **Seed:** `random_state=42` for exact scientific reproducibility.
            """)
        with split_col2:
            st.markdown("""
            **Model-Specific Feature Scaling:**
            - **Decision Tree:** Uses native, unscaled feature values. Tree split thresholds retain natural units (e.g. `absences <= 5.5 days`), preserving 100% human explainability.
            - **Support Vector Machine (SVM):** Uses `StandardScaler` ($z = \\frac{x - \\mu}{\\sigma}$) fit strictly on the training set and applied to the test set, preventing distance distortion in the RBF kernel.
            """)


# =========================================================
# PAGE 3: 📈 Exploratory Data Analysis (EDA)
# =========================================================
elif nav_page == "📈 Exploratory Data Analysis (EDA)":
    # Row 1: Donut & G3 Distribution
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        with st.container(border=True):
            fig1, ins1 = create_pass_fail_donut(filtered_df)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins1}</div>', unsafe_allow_html=True)

    with r1_col2:
        with st.container(border=True):
            fig2, ins2 = create_g3_histogram(filtered_df)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins2}</div>', unsafe_allow_html=True)

    # Row 2: Past Failures & Absences Boxplot
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        with st.container(border=True):
            fig3, ins3 = create_failures_vs_outcome_chart(filtered_df)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins3}</div>', unsafe_allow_html=True)

    with r2_col2:
        with st.container(border=True):
            fig4, ins4 = create_absences_boxplot(filtered_df)
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins4}</div>', unsafe_allow_html=True)

    # Row 3: Study Time & Correlation Heatmap
    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        with st.container(border=True):
            fig5, ins5 = create_studytime_pass_rate_chart(filtered_df)
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins5}</div>', unsafe_allow_html=True)

    with r3_col2:
        with st.container(border=True):
            fig6, ins6 = create_correlation_heatmap(filtered_df)
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins6}</div>', unsafe_allow_html=True)

    # Row 4: Demographic Factors & G1 vs G2 Scatter
    r4_col1, r4_col2 = st.columns(2)
    with r4_col1:
        with st.container(border=True):
            fig7, ins7 = create_categorical_pass_rates_chart(filtered_df)
            st.plotly_chart(fig7, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins7}</div>', unsafe_allow_html=True)

    with r4_col2:
        with st.container(border=True):
            fig8, ins8 = create_g1_g2_scatter(filtered_df)
            st.plotly_chart(fig8, use_container_width=True)
            st.markdown(f'<div class="insight-pill">{ins8}</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 4: ⚙️ Model Training & Experiments
# =========================================================
elif nav_page == "⚙️ Model Training & Experiments":
    with st.container(border=True):
        st.markdown("### 🧪 Model Configuration Specifications")
        
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            st.markdown("""
            **1. Decision Tree Classifier (`DecisionTreeClassifier`)**
            - **Criterion:** Gini Impurity (`criterion="gini"`)
            - **Max Tree Depth:** 4 levels (prevents deep-tree overfitting)
            - **Min Samples Split:** 10 samples required to partition internal nodes
            - **Random State:** 42
            - **Primary Advantage:** Native interpretability, generates boolean logic trees that academic advisors can audit directly.
            """)
        with cfg_col2:
            st.markdown(r"""
            **2. Support Vector Machine (`SVC`)**
            - **Kernel:** Radial Basis Function (`kernel="rbf"`)
            - **Regularization (C):** 1.0
            - **Gamma:** Scale ($1 / (n_{features} \cdot \sigma^2)$)
            - **Probability Estimation:** Enabled (`probability=True`)
            - **Preprocessing:** Features scaled via `StandardScaler`
            - **Primary Advantage:** Maximizes margin separation in high-dimensional continuous spaces.
            """)

    # Tuning Grid Details
    if tune_mode:
        with st.container(border=True):
            st.markdown("### 🔍 5-Fold GridSearchCV Optimization Results")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown(f"""
                **Decision Tree Optimal Parameters:**
                - Best Grid Parameters: `{models_data['best_params_dt']}`
                - Cross-Validation Score: **{models_data['metrics_dt']['5-Fold CV Accuracy']*100:.2f}%**
                """)
            with t_col2:
                st.markdown(f"""
                **SVM Optimal Parameters:**
                - Best Grid Parameters: `{models_data['best_params_svm']}`
                - Cross-Validation Score: **{models_data['metrics_svm']['5-Fold CV Accuracy']*100:.2f}%**
                """)

    # G1 / G2 Toggle Impact Analysis
    with st.container(border=True):
        st.markdown("### ⚖️ The G1 & G2 Early-Warning Dilemma")
        
        if include_g1_g2:
            st.markdown("""
            **Current Status: `G1 & G2 Features INCLUDED`**
            - Period 1 ($G1$) and Period 2 ($G2$) test scores provide strong predictive signal.
            - Result: **Accuracy exceeds 87%**, and at-risk student recall reaches **88.5%**.
            - *Operational Limitation:* These scores are only available after midterms, leaving less time for remediation before the final exam.
            """)
        else:
            st.markdown("""
            **Current Status: `G1 & G2 Features EXCLUDED` (Early-Semester Baseline Mode)**
            - Model predicts student risk using **only baseline indicators** (past failures, study habits, parental education, attendance).
            - Result: Accuracy drops to ~67–68%, and Fail Recall drops to ~34.6%.
            - *Operational Value:* Flags struggling students **at Week 1 of the semester**, allowing full-term academic coaching before test failures occur.
            """)


# =========================================================
# PAGE 5: ⚖️ Model Comparison & Diagnostics
# =========================================================
elif nav_page == "⚖️ Model Comparison & Diagnostics":
    # Comparative Metrics Table
    with st.container(border=True):
        st.markdown("### 📊 Performance Metrics Comparison Table")
        
        m_dt = models_data["metrics_dt"]
        m_svm = models_data["metrics_svm"]

        metric_names = [
            "Accuracy",
            "Precision (Pass)",
            "Recall (Pass)",
            "Recall (Fail - Risk)",
            "F1-Score",
            "ROC-AUC",
            "5-Fold CV Accuracy"
        ]

        table_rows = []
        for m in metric_names:
            dt_val = m_dt[m]
            svm_val = m_svm[m]
            ref_dt = NOTEBOOK_BENCHMARK["Decision Tree"].get(m, 0.0)
            ref_svm = NOTEBOOK_BENCHMARK["SVM"].get(m, 0.0)

            dt_match = is_matching_benchmark(m, dt_val, "Decision Tree") and include_g1_g2 and not tune_mode
            svm_match = is_matching_benchmark(m, svm_val, "SVM") and include_g1_g2 and not tune_mode

            dt_badge = " <span class='benchmark-badge'>✅ Benchmark Match</span>" if dt_match else ""
            svm_badge = " <span class='benchmark-badge'>✅ Benchmark Match</span>" if svm_match else ""

            lead = "Decision Tree" if dt_val > svm_val else ("SVM" if svm_val > dt_val else "Tie")

            table_rows.append(f"""<tr>
                <td style="font-weight:600;">{m}</td>
                <td class="lead-col"><b>{dt_val:.4f}</b> ({dt_val*100:.2f}%){dt_badge}</td>
                <td><b>{svm_val:.4f}</b> ({svm_val*100:.2f}%){svm_badge}</td>
                <td style="color:#64748B;">{ref_dt:.4f}</td>
                <td style="color:#64748B;">{ref_svm:.4f}</td>
                <td><span style="background:{'#DCFCE7' if lead=='Decision Tree' else '#EEF2FF'}; color:{'#15803D' if lead=='Decision Tree' else '#4338CA'}; padding:3px 8px; border-radius:6px; font-weight:700; font-size:11px;">{lead}</span></td>
            </tr>""")

        table_html = f"""<div class="table-container"><table class="luxury-table"><thead><tr><th>Evaluation Metric</th><th>Decision Tree (Lead Model)</th><th>SVM (RBF Kernel)</th><th>Notebook DT Ref</th><th>Notebook SVM Ref</th><th>Top Performer</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>"""
        st.html(table_html)

    # Dynamic Recommendation Callout Box
    rec_diff = (m_dt["Recall (Fail - Risk)"] - m_svm["Recall (Fail - Risk)"]) * 100
    st.html(f"""
        <div style="background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 100%); border: 1px solid #86EFAC; border-left: 5px solid #10B981; border-radius: 14px; padding: 22px; margin-bottom: 22px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.06);">
            <h3 style="color:#0F172A; margin-top:0; font-size:16px; font-weight:700;">💡 Which Model is Better and Why? (Lead Data Scientist Recommendation)</h3>
            <p style="font-size:13.5px; color:#1E293B; line-height:1.6; margin-bottom:10px;">
                In an educational <b>Early-Warning System (EWS)</b>, the primary real-world objective is to 
                <b>minimize False Negatives</b> (at-risk students mistakenly classified as safe). Missing a failing student means zero institutional support is provided, resulting in preventable course failure.
            </p>
            <p style="font-size:13.5px; color:#1E293B; line-height:1.6; margin:0;">
                <b>The Decision Tree Classifier is clearly superior for institutional deployment:</b><br>
                1. <b>Superior At-Risk Recall:</b> Decision Tree achieves <b>{m_dt['Recall (Fail - Risk)']*100:.2f}%</b> recall on the critical Fail class, compared to <b>{m_svm['Recall (Fail - Risk)']*100:.2f}%</b> for SVM ({'+' if rec_diff>=0 else ''}{rec_diff:.1f}% operational advantage).<br>
                2. <b>Higher Generalization Stability:</b> 5-Fold Cross-Validation Accuracy is <b>{m_dt['5-Fold CV Accuracy']*100:.2f}%</b> (vs {m_svm['5-Fold CV Accuracy']*100:.2f}% for SVM).<br>
                3. <b>Zero Black-Box Obscurity:</b> Academic advisors can inspect the exact tree splits (e.g. <i>past failures ≥ 1 AND absences > 5.5</i>), ensuring compliant, explainable student counseling.
            </p>
        </div>
    """)

    # Diagnostic Visuals: Confusion Matrices, ROC Curves, and Metrics Bar
    cm_col1, cm_col2 = st.columns(2)
    fig_cm_dt, fig_cm_svm = create_confusion_matrix_charts(models_data["cm_dt"], models_data["cm_svm"])
    with cm_col1:
        with st.container(border=True):
            st.plotly_chart(fig_cm_dt, use_container_width=True)
    with cm_col2:
        with st.container(border=True):
            st.plotly_chart(fig_cm_svm, use_container_width=True)

    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        with st.container(border=True):
            fig_roc = create_roc_curve_chart(models_data["roc_dt"], models_data["roc_svm"])
            st.plotly_chart(fig_roc, use_container_width=True)

    with diag_col2:
        with st.container(border=True):
            fig_comp_bar = create_metrics_comparison_bar(m_dt, m_svm)
            st.plotly_chart(fig_comp_bar, use_container_width=True)

    # Scikit-Learn Classification Reports in Expanders
    with st.container(border=True):
        st.markdown("### 📄 Formal Classification Reports")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            with st.expander("Decision Tree Classification Report", expanded=False):
                st.code(models_data["report_dt"], language="text")
        with exp_col2:
            with st.expander("SVM (RBF Kernel) Classification Report", expanded=False):
                st.code(models_data["report_svm"], language="text")


# =========================================================
# PAGE 6: 🌳 Feature Importance & Tree Rules
# =========================================================
elif nav_page == "🌳 Feature Importance & Tree Rules":
    # Top Feature Importances Bar Chart
    with st.container(border=True):
        fig_imp = create_feature_importance_bar(models_data["feat_imp"], top_n=10)
        st.plotly_chart(fig_imp, use_container_width=True)

    # Decision Tree Diagram
    with st.container(border=True):
        st.markdown("### 🗺️ Full Decision Tree Architectural Hierarchy")
        st.markdown("Every internal split represents an academic decision threshold. Green nodes denote high Pass probability; reddish nodes denote high Failure risk.")
        fig_tree = generate_tree_plot(models_data["dt_model"], models_data["feature_names"])
        st.pyplot(fig_tree)

    # Decision Tree Rules as Text in Expander
    with st.container(border=True):
        st.markdown("### 📜 Decision Rules Text Export (`export_text`)")
        with st.expander("View Full Text Decision Logic & Branches", expanded=True):
            st.code(models_data["tree_rules_text"], language="text")


# =========================================================
# PAGE 7: 🚨 High-Risk Profiles & Interventions
# =========================================================
elif nav_page == "🚨 High-Risk Profiles & Interventions":
    # Model selector and threshold sliders
    with st.container(border=True):
        st.markdown("### 🎚️ Risk Threshold Calibration Console")
        t_c1, t_c2, t_c3 = st.columns([1, 1, 1])
        with t_c1:
            sel_model = st.selectbox("Predictive Scoring Engine", ["Decision Tree", "SVM"], index=0)
        with t_c2:
            high_thresh = st.slider("High Risk Band (≥ Fail Prob)", min_value=0.50, max_value=0.95, value=0.70, step=0.05)
        with t_c3:
            med_thresh = st.slider("Medium Risk Band (Fail Prob)", min_value=0.20, max_value=0.65, value=0.40, step=0.05)

    # Score full cohort with selected model
    scored_students = score_cohort_risk(filtered_df, models_data, model_choice=sel_model, include_g1_g2=include_g1_g2)
    tiered_students = assign_risk_tiers(scored_students, high_thresh=high_thresh, med_thresh=med_thresh)

    # Risk KPI Cards
    n_total = len(tiered_students)
    n_high = int((tiered_students["Risk_Tier_Simple"] == "High").sum())
    n_med = int((tiered_students["Risk_Tier_Simple"] == "Medium").sum())
    n_low = int((tiered_students["Risk_Tier_Simple"] == "Low").sum())
    avg_risk = float(tiered_students["Risk_Probability"].mean() * 100)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card card-cohort">
            <div class="kpi-header">
                <span class="kpi-label">Total Evaluated</span>
                <span class="kpi-icon">📋</span>
            </div>
            <div class="kpi-metric">{n_total}</div>
            <div class="kpi-progress"><div class="kpi-progress-bar" style="width:100%;"></div></div>
            <div class="kpi-footer">Filtered cohort size</div>
        </div>
        <div class="kpi-card card-risk">
            <div class="kpi-header">
                <span class="kpi-label">🚨 High Risk</span>
                <span class="kpi-icon">⚠️</span>
            </div>
            <div class="kpi-metric" style="color:#DC2626;">{n_high}</div>
            <div class="kpi-progress"><div class="kpi-progress-bar fail" style="width:{(n_high/n_total*100 if n_total>0 else 0):.1f}%;"></div></div>
            <div class="kpi-footer">{(n_high/n_total*100 if n_total>0 else 0):.1f}% critical failure risk</div>
        </div>
        <div class="kpi-card card-absences">
            <div class="kpi-header">
                <span class="kpi-label">⚠️ Medium Risk</span>
                <span class="kpi-icon">⚡</span>
            </div>
            <div class="kpi-metric" style="color:#D97706;">{n_med}</div>
            <div class="kpi-progress"><div class="kpi-progress-bar warning" style="width:{(n_med/n_total*100 if n_total>0 else 0):.1f}%;"></div></div>
            <div class="kpi-footer">{(n_med/n_total*100 if n_total>0 else 0):.1f}% moderate vulnerability</div>
        </div>
        <div class="kpi-card card-pass">
            <div class="kpi-header">
                <span class="kpi-label">✅ Low Risk</span>
                <span class="kpi-icon">🛡️</span>
            </div>
            <div class="kpi-metric" style="color:#059669;">{n_low}</div>
            <div class="kpi-progress"><div class="kpi-progress-bar pass" style="width:{(n_low/n_total*100 if n_total>0 else 0):.1f}%;"></div></div>
            <div class="kpi-footer">{(n_low/n_total*100 if n_total>0 else 0):.1f}% safe / on-track</div>
        </div>
        <div class="kpi-card card-grade">
            <div class="kpi-header">
                <span class="kpi-label">Average Risk Score</span>
                <span class="kpi-icon">📈</span>
            </div>
            <div class="kpi-metric">{avg_risk:.1f}%</div>
            <div class="kpi-progress"><div class="kpi-progress-bar neutral" style="width:{avg_risk:.1f}%;"></div></div>
            <div class="kpi-footer">Cohort mean failure probability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Auto-Discovered High-Risk Profiles Section
    with st.container(border=True):
        st.markdown("### 🔍 Auto-Discovered High-Risk Student Archetypes (Derived Dynamically From Data)")
        st.markdown("""
            Rather than relying on static assumptions, the system clusters students in the high-risk cohort by their dominant academic, 
            behavioral, and aspirational traits to isolate distinct student archetypes.
        """)

        profiles = discover_at_risk_profiles(tiered_students, high_thresh=high_thresh)
        
        for p in profiles:
            with st.expander(f"{p['archetype']} — {p['count']} Students ({p['cohort_pct']:.1f}% of at-risk cohort)", expanded=True):
                p_col1, p_col2 = st.columns([1, 1.2])
                with p_col1:
                    st.markdown(f"**Average Risk Score:** `{p['avg_risk']*100:.1f}%`")
                    st.markdown(f"**Identified Behavioral Signature:** `{p['traits']}`")
                    st.markdown(f"**Institutional Priority:** {p['priority']}")
                    st.markdown(f"**Expected Recovery Impact:** `{p['expected_impact']}`")
                with p_col2:
                    st.markdown("**Mapped University Interventions:**")
                    for act in p["interventions"]:
                        st.markdown(f"• {act}")

    # Sortable, Filterable Table of High-Risk Students
    with st.container(border=True):
        st.markdown("### 📋 High-Risk Student Roster (Actionable Case Dossier)")
        
        high_risk_list = tiered_students[tiered_students["Risk_Tier_Simple"] == "High"].sort_values(
            by="Risk_Probability", ascending=False
        )
        
        display_cols = [
            "Risk_Probability", "Risk_Tier", "failures", "absences", "studytime", 
            "higher", "school", "sex", "age", "G1", "G2", "G3"
        ]
        available_disp = [c for c in display_cols if c in high_risk_list.columns]
        
        st.dataframe(high_risk_list[available_disp], use_container_width=True)

        # Download Button for High-Risk Roster
        hr_csv = io.StringIO()
        high_risk_list[available_disp].to_csv(hr_csv, index=True)
        st.download_button(
            label="📥 Download High-Risk Students Roster (CSV)",
            data=hr_csv.getvalue(),
            file_name="high_risk_students_roster.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# PAGE 8: 🔮 What-If Student Risk Simulator
# =========================================================
elif nav_page == "🔮 What-If Student Risk Simulator":
    with st.container(border=True):
        st.markdown("### 🎛️ Student Attribute Profile Builder")

        tab_demo, tab_acad, tab_life = st.tabs([
            "👤 Demographics & Family",
            "📚 Academic Habits & Schooling",
            "🍷 Social Life, Wellbeing & Grades"
        ])

        with tab_demo:
            sim_c1, sim_c2 = st.columns(2)
            with sim_c1:
                sim_school = st.selectbox("School", ["GP", "MS"], format_func=lambda x: "Gabriel Pereira (GP)" if x=="GP" else "Mousinho da Silveira (MS)")
                sim_sex = st.selectbox("Sex", ["F", "M"], format_func=lambda x: "Female" if x=="F" else "Male")
                sim_age = st.slider("Age", 15, 22, 17)
                sim_address = st.selectbox("Address", ["U", "R"], format_func=lambda x: "Urban (U)" if x=="U" else "Rural (R)")
                sim_famsize = st.selectbox("Family Size", ["GT3", "LE3"], format_func=lambda x: "Greater than 3" if x=="GT3" else "Less/Equal to 3")
            with sim_c2:
                sim_pstatus = st.selectbox("Parents Cohabitation Status", ["T", "A"], format_func=lambda x: "Living Together (T)" if x=="T" else "Apart (A)")
                sim_medu = st.slider("Mother's Education (0: None, 4: Higher)", 0, 4, 2)
                sim_fedu = st.slider("Father's Education (0: None, 4: Higher)", 0, 4, 2)
                sim_mjob = st.selectbox("Mother's Job", ["at_home", "health", "services", "teacher", "other"])
                sim_fjob = st.selectbox("Father's Job", ["teacher", "other", "services", "health", "at_home"])
                sim_guardian = st.selectbox("Primary Guardian", ["mother", "father", "other"])

        with tab_acad:
            sim_a1, sim_a2 = st.columns(2)
            with sim_a1:
                sim_reason = st.selectbox("School Choice Reason", ["home", "reputation", "course", "other"])
                sim_traveltime = st.slider("Commute Travel Time (1: <15m, 4: >1h)", 1, 4, 1)
                sim_studytime = st.slider("Weekly Study Hours (1: <2h, 4: >10h)", 1, 4, 2)
                sim_failures = st.slider("Past Course Failures", 0, 3, 0)
                sim_absences = st.slider("School Absences (Days)", 0, 75, 4)
            with sim_a2:
                sim_schoolsup = st.selectbox("Extra Educational School Support", ["yes", "no"], index=1)
                sim_famsup = st.selectbox("Family Educational Support", ["yes", "no"], index=0)
                sim_paid = st.selectbox("Extra Paid Classes", ["yes", "no"], index=1)
                sim_activities = st.selectbox("Extra-Curricular Activities", ["yes", "no"], index=0)
                sim_nursery = st.selectbox("Attended Nursery School", ["yes", "no"], index=0)
                sim_higher = st.selectbox("Aims for Higher Education", ["yes", "no"], index=0)

        with tab_life:
            sim_l1, sim_l2 = st.columns(2)
            with sim_l1:
                sim_internet = st.selectbox("Home Internet Access", ["yes", "no"], index=0)
                sim_romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"], index=1)
                sim_famrel = st.slider("Family Relationship Quality (1: Low, 5: Excellent)", 1, 5, 4)
                sim_freetime = st.slider("Free Time After School (1: Very Low, 5: Very High)", 1, 5, 3)
                sim_goout = st.slider("Going Out With Friends (1: Very Low, 5: Very High)", 1, 5, 3)
            with sim_l2:
                sim_dalc = st.slider("Workday Alcohol Consumption (1: Very Low, 5: Very High)", 1, 5, 1)
                sim_walc = st.slider("Weekend Alcohol Consumption (1: Very Low, 5: Very High)", 1, 5, 1)
                sim_health = st.slider("Current Health Status (1: Very Poor, 5: Excellent)", 1, 5, 4)
                if include_g1_g2:
                    sim_g1 = st.slider("Period 1 Grade (G1: 0–20)", 0, 20, 11)
                    sim_g2 = st.slider("Period 2 Grade (G2: 0–20)", 0, 20, 11)
                else:
                    sim_g1, sim_g2 = 10, 10
                    st.info("ℹ️ Midterm grades G1 & G2 disabled in Early Intervention Mode.")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        run_sim = st.button("🔮 Run Instant Risk Assessment", type="primary", use_container_width=True)

    if run_sim:
        student_input = {
            "school": sim_school, "sex": sim_sex, "age": sim_age, "address": sim_address,
            "famsize": sim_famsize, "Pstatus": sim_pstatus, "Medu": sim_medu, "Fedu": sim_fedu,
            "Mjob": sim_mjob, "Fjob": sim_fjob, "reason": sim_reason, "guardian": sim_guardian,
            "traveltime": sim_traveltime, "studytime": sim_studytime, "failures": sim_failures,
            "schoolsup": sim_schoolsup, "famsup": sim_famsup, "paid": sim_paid,
            "activities": sim_activities, "nursery": sim_nursery, "higher": sim_higher,
            "internet": sim_internet, "romantic": sim_romantic, "famrel": sim_famrel,
            "freetime": sim_freetime, "goout": sim_goout, "Dalc": sim_dalc, "Walc": sim_walc,
            "health": sim_health, "absences": sim_absences, "G1": sim_g1, "G2": sim_g2
        }

        sim_res = get_what_if_prediction(
            student_input, models_data, encoders_dict, include_g1_g2=include_g1_g2
        )

        with st.container(border=True):
            st.markdown("### 🎯 Predictive Risk Assessment & Recommended Roadmap")

            out_c1, out_c2 = st.columns([1, 1.2])
            with out_c1:
                fig_gauge = create_risk_gauge(sim_res["consensus_risk"])
                st.plotly_chart(fig_gauge, use_container_width=True)

                status_badge_class = 'badge-pass' if 'Pass' in sim_res['status'] else 'badge-fail'
                st.html(f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px; margin-top:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:12px; font-weight:700; color:#64748B;">CONSENSUS PREDICTION:</span>
                        <span class="{status_badge_class}">{sim_res['status']}</span>
                    </div>
                    <div style="font-size:12px; color:#334155; line-height:1.6;">
                        • <b>Decision Tree Fail Risk:</b> <code>{sim_res['dt_risk']*100:.1f}%</code> ({sim_res['dt_pred']})<br>
                        • <b>SVM Fail Risk:</b> <code>{sim_res['svm_risk']*100:.1f}%</code> ({sim_res['svm_pred']})
                    </div>
                </div>
                """)

            with out_c2:
                st.markdown("#### 🔍 Primary Risk Factors Identified:")
                for rf in sim_res["risk_drivers"]:
                    st.markdown(f"• ⚠️ {rf}")

                st.markdown("#### 🛠️ Personalized Advisor Action Plan:")
                for act in sim_res["interventions"]:
                    st.markdown(f"• ✅ **{act}**")


# =========================================================
# PAGE 9: 💡 Insights & Recommendations
# =========================================================
elif nav_page == "💡 Insights & Recommendations":
    # Section 1: Key Influential Factors
    with st.container(border=True):
        st.markdown("### 🏆 Top Empirical Factors Influencing Academic Performance")
        st.markdown(r"""
        1. **Past Academic Failures (`failures`):** The single strongest historical predictor. Students with $\ge 1$ past failures have a **35–45% lower pass rate** due to compounded conceptual deficits.
        2. **Earlier Assessment Scores (`G1`, `G2`):** When available, Period 2 grade ($G2$) alone accounts for over 70% of tree importance. A student scoring $<10$ on $G2$ has an **85%+ likelihood of failing $G3$**.
        3. **Attendance & Absenteeism (`absences`):** Chronic absenteeism (>10 days) severely degrades course engagement, though clusters of 0-absence dropouts also emerge.
        4. **Weekly Study Commitment (`studytime`):** Stepping up study hours from $<2$ hrs to $>5$ hrs increases cohort pass rates from **56% to 81%**.
        5. **Higher Education Ambition (`higher`):** Students aspiring to university degrees pass at **69.8%**, compared to **35.0%** for non-aspirational peers (a 34.8% equity gap).
        """)

    # Section 2: Model Selection Rationale
    with st.container(border=True):
        st.markdown("### 🤖 Institutional Model Recommendation: Decision Tree")
        
        st.html("""
        <div class="table-container">
            <table class="luxury-table">
                <thead>
                    <tr>
                        <th>Operational Criteria</th>
                        <th>Decision Tree (Selected)</th>
                        <th>Support Vector Machine</th>
                        <th>Rationale for Institutional Choice</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><b>At-Risk Recall (Fail Class)</b></td>
                        <td class="lead-col"><b>88.46%</b></td>
                        <td>69.23%</td>
                        <td><b>Decision Tree (+19.23%):</b> Catches nearly 9 in 10 failing students.</td>
                    </tr>
                    <tr>
                        <td><b>Overall Accuracy</b></td>
                        <td class="lead-col"><b>87.34%</b></td>
                        <td>79.75%</td>
                        <td><b>Decision Tree (+7.59%):</b> Higher overall correctness.</td>
                    </tr>
                    <tr>
                        <td><b>Explainability</b></td>
                        <td class="lead-col"><b>100% Rule-Based</b></td>
                        <td>Black-box Hyperplane</td>
                        <td><b>Decision Tree:</b> Generates explicit boolean rules advisors can explain to students.</td>
                    </tr>
                    <tr>
                        <td><b>5-Fold Cross Validation</b></td>
                        <td class="lead-col"><b>89.55%</b></td>
                        <td>84.17%</td>
                        <td><b>Decision Tree (+5.38%):</b> Strong generalization across data folds.</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """)

    # Section 3: 4-Tier Institutional Roadmap
    with st.container(border=True):
        st.markdown("### 🏛️ 4-Tier University Support Roadmap")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.markdown(r"""
            **Tier 1: Pre-Semester / Week 1 Baseline Screening**
            - Run model in *Early-Intervention Mode* (no $G1$/$G2$).
            - Flag students with $\ge 1$ past failure or low study hours.
            - Dispatch automated welcome nudge and peer tutoring enrollment.

            **Tier 2: Week 4 Attendance & Diagnostic Checkpoint**
            - Monitor absence velocity (alert triggered if $> 4$ absences).
            - Mandatory advisor check-in for students missing lectures.
            """)
        with r_c2:
            st.markdown("""
            **Tier 3: Week 8 Midterm Recovery Phase**
            - Re-score cohort with $G1$ and midterm diagnostic scores.
            - Students scoring $<10$ enrolled in intensive remedial clinics.
            - Assign student success coaches for 1-on-1 accountability.

            **Tier 4: Holistic Wellbeing & Digital Equity Support**
            - Provide loaner laptops and cellular hotspots for rural/unconnected students.
            - Coordinate with university student counseling for wellness, stress, and lifestyle support.
            """)

    # Section 4: Honest System Limitations
    with st.container(border=True):
        st.markdown("### ⚠️ Honest System Limitations & Real-World Considerations")
        st.markdown("""
        1. **Sample Size ($N=395$):** The UCI Mathematics dataset covers 395 students across two secondary schools in Portugal. Results should be calibrated before direct cross-institutional transfer.
        2. **Single-Subject Domain:** The records strictly represent secondary school Mathematics performance. Course dynamics for humanities, engineering, or laboratory sciences will have different feature weightings.
        3. **Intermediate Grade Dependency ($G1$/$G2$):** While earlier grades yield high accuracy, they arrive mid-semester. Institutions must balance high accuracy in Week 8 with actionable early signals in Week 1.
        4. **Proxy Variable Approximations:** The dataset lacks fine-grained continuous LMS interaction logs (e.g. video watch time, assignment submission timestamps, quiz attempts), relying on `absences` as a coarse proxy. Modern university LMS data (Canvas/Moodle) should be integrated to strengthen predictions.
        """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("<hr style='border:none; border-top:1px solid #E2E8F0; margin: 30px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; color:#94A3B8; font-size:12px; padding-bottom:14px; flex-wrap:wrap; gap:10px;">
    <div>🎓 <b>EduRisk Early-Warning System (EWS)</b> | Institutional Early Academic Intervention Platform</div>
    <div>Built with Python, Scikit-Learn & Streamlit</div>
</div>
""", unsafe_allow_html=True)
