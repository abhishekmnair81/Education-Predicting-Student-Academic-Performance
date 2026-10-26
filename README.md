# 🎓 Student Academic Performance Predictor & Early-Warning System (EWS)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F51B5.svg)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning Early-Warning System (EWS) and interactive diagnostic dashboard developed to predict university student academic performance and identify **at-risk students ($G3 < 10$)** weeks prior to final examinations. Utilizing real-world academic, behavioral, and demographic datasets (`student-mat.csv`), this project provides real-time reactive visualizations, comparative model evaluations (Decision Tree vs. SVM), transparent feature importance trees, automated high-risk profile clustering, and an institutional 4-tier intervention framework.

---

## 📌 Executive Summary

Traditional institutional evaluations are **reactive**—identifying failing students only after final exams when it is too late to intervene. This project implements a **proactive Early-Warning System** designed to:
1. **Prevent Data Leakage:** Exclude final exam outputs ($G3$) from features so predictions can be made prior to final testing.
2. **Prioritize High At-Risk Recall:** Optimize recall on the failing student class (**Class 0**) to minimize the institutional cost of False Negatives (unidentified struggling students).
3. **Provide Transparent Triage:** Leverage an explainable **Decision Tree Classifier** whose boolean rules advisors can inspect and trust.
4. **Automate High-Risk Profiling:** Group vulnerable students by distinct behavioral signatures (e.g. chronic past failures, high absenteeism, aspiration gaps) rather than relying on static labels.
5. **Simulate What-If Scenarios:** Provide a real-time risk simulator with customized advisor action plans.

---

## 📁 Repository Structure

```
PADV_ACT/
│
├── app.py                      # Main Streamlit reactive dashboard application
├── utils/
│   ├── __init__.py             # Utils package initialization
│   ├── data_loader.py          # Data ingestion, schema validation, leakage prevention & filtering
│   ├── model_trainer.py        # Model training (DT & SVM), 5-fold CV, GridSearchCV, benchmark badges
│   ├── visualizations.py       # Plotly charts with unified theme & dynamic calculated insights
│   └── risk_analyzer.py        # Cohort risk scoring, profile discovery & what-if simulator
│
├── student-mat.csv             # UCI Student Performance (Mathematics) dataset (395 rows, 33 cols)
├── Education.ipynb             # Research notebook containing initial data exploration & model prototyping
├── requirements.txt            # Frozen production dependencies
└── README.md                   # System documentation & run instructions
```

---

## 📊 Dataset Overview (`student-mat.csv`)

The dataset comprises 395 student instances with 33 attributes spanning academic history, demographic factors, social habits, and family background (UCI Machine Learning Repository, Mathematics).

### Target Variable Engineering
$$\text{Target} = \begin{cases} 1 \quad (\text{Pass}), & \text{if } G3 \ge 10 \quad (265 \text{ students / } 67.09\%) \\ 0 \quad (\text{Fail / At-Risk}), & \text{if } G3 < 10 \quad (130 \text{ students / } 32.91\%) \end{cases}$$

### Proxy Variables Note
This dataset does not record direct continuous attendance percentages or daily assignment grades. Standard proxy mappings are utilized and highlighted throughout the dashboard:
* **Attendance Proxy:** `absences` (school absences in days, 0–75)
* **Earlier Assessment Scores:** `G1` (Period 1 test grade) & `G2` (Period 2 test grade)
* **Study Commitment:** `studytime` (weekly study hours categorized 1–4)

---

## 🚀 Quick Start & Run Instructions

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12
* Git and virtual environment tools (`venv` or `conda`)

### 2. Setup Virtual Environment
```bash
# Clone the repository (or navigate to workspace directory)
cd PADV_ACT

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
The application will open in your default browser at **`http://localhost:8501`**.

---

## 🖥️ Dashboard Page Tour & Features

| # | Page / Tab | Key Features & Output |
| :--- | :--- | :--- |
| **1** | **📊 Executive Overview** | High-level cohort health cards, Pass vs Fail donut distribution, G3 final grade histogram, and institutional Early-Warning architecture workflow. |
| **2** | **🧹 Data & Preprocessing** | Dataset shape ($395 \times 33$), missing values verification ($0$), duplicate check ($0$), raw preview table, data leakage prevention protocol, and before/after label encoding preview for 17 categorical variables. |
| **3** | **📈 Exploratory Data Analysis (EDA)** | 8 reactive Plotly charts (Pass/Fail donut, G3 histogram with pass line, failures vs outcome, absences boxplot, study time vs pass rate, correlation heatmap, demographic equity gaps, and G1 vs G2 scatter) with dynamic auto-generated insights under each chart. |
| **4** | **⚙️ Model Training & Experiments** | Model architectures, hyperparameter configurations, Default vs Tuned (5-Fold GridSearchCV) comparison, and G1/G2 feature inclusion toggle with performance impact analysis. |
| **5** | **⚖️ Model Comparison & Diagnostics** | Comprehensive metrics table (Accuracy, Precision, Pass Recall, Fail Recall, F1, ROC-AUC, 5-Fold CV), green "✅ Notebook Match" badges, side-by-side confusion matrices, ROC curves, and senior data scientist recommendation. |
| **6** | **🌳 Feature Importance & Tree Rules** | Top 10 feature importances horizontal bar chart, full architectural `plot_tree` diagram, and exportable decision logic rules (`export_text`). |
| **7** | **🚨 High-Risk Profiles & Interventions** | Dynamic risk threshold sliders (High ≥ 70%, Medium 40–70%, Low < 40%), risk KPI summary, auto-discovered high-risk student profiles, and actionable student roster with one-click CSV export. |
| **8** | **🔮 What-If Student Risk Simulator** | Interactive sliders and dropdowns for all 32 student features, real-time dual-model prediction, animated risk speedometer gauge, top risk driver analysis, and personalized advisor action checklist. |
| **9** | **💡 Insights & Recommendations** | Top empirical academic drivers, model selection synthesis, 4-tier institutional intervention roadmap, and honest system limitations. |

---

## 📈 Model Performance & Comparative Benchmark

Models were trained with an **80/20 Stratified Train-Test Split ($N_{train}=316, N_{test}=79$)** with `random_state=42`. Decision Tree unscaled features; SVM standard-scaled via `StandardScaler`.

| Evaluation Metric | Decision Tree (depth=4) | SVM Classifier (RBF) | Lead Model | Notebook Benchmark Status |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **87.34%** | 79.75% | **Decision Tree (+7.59%)** | ✅ Matches Notebook (`0.8734` vs `0.7975`) |
| **Precision (Pass Class)** | **93.88%** | 84.91% | **Decision Tree (+8.97%)** | ✅ Matches Notebook (`0.9388` vs `0.8491`) |
| **Recall (Pass Class)** | 86.79% | 84.91% | **Decision Tree (+1.88%)** | ✅ Matches Notebook (`0.8679` vs `0.8491`) |
| **Recall (Fail - At-Risk Class)** | **88.46%** | 69.23% | **Decision Tree (+19.23%)** | ✅ Matches Notebook (`0.8846` vs `0.6923`) |
| **F1-Score** | **0.9020** | 0.8491 | **Decision Tree (+0.0529)** | ✅ Matches Notebook (`0.9020` vs `0.8491`) |
| **ROC-AUC Score** | **0.9267** | 0.8919 | **Decision Tree (+0.0348)** | ✅ Matches Notebook (`0.9267` vs `0.8919`) |
| **5-Fold CV Accuracy** | **89.55%** | 84.17% | **Decision Tree (+5.38%)** | ✅ Matches Notebook (`0.8955` vs `0.8417`) |

### Why Decision Tree is Selected for Institutional Deployment:
1. **Critical At-Risk Recall (88.46%):** Decision Tree catches almost 9 out of 10 failing students, whereas SVM misses over 30% of at-risk students. In academic triage, a False Negative (failing student unassisted) is far costlier than a False Positive (safe student given tutoring).
2. **Transparent Decision Rules:** Deans and advisors can view the exact boolean decision paths (e.g. `failures > 0.5 AND absences > 5.5`), ensuring explainable, non-discriminatory student interventions.
3. **Generalization Stability:** Exhibits a **89.55% 5-Fold Cross-Validation Accuracy**, proving absence of fold-level overfitting.

---

## 🏛️ 4-Tier Institutional Intervention Framework

```
[ Week 1: Baseline Screening ] ➔ Automated early risk triage before midterms
            │
[ Week 4: Diagnostic Checkpoint ] ➔ Absence velocity alerts & 1-on-1 advisor check-ins
            │
[ Week 8: Midterm Recovery ] ➔ Remedial clinics & peer tutoring for G1/G2 < 10
            │
[ Holistic Retention Support ] ➔ Digital equity loaner laptops & student counseling referrals
```

1. **Chronic Academic Deficit:** Auto-enroll students with past failures into mandatory peer tutoring and concept foundation sessions.
2. **Attendance Disengagement:** Automated SMS alerts and advisor outreach triggered when absences exceed 6 days.
3. **Study-Skill Deficit:** Academic time-management workshops and minimum 4-hr/wk structured library blocks.
4. **Digital & Commuter Equity:** Subsidized mobile Wi-Fi hotspots, recorded lectures, and transit assistance for rural students.

---

## ⚠️ Real-World Limitations

1. **Sample Size ($N=395$):** Dataset represents two secondary schools in Portugal. Cross-institutional deployment should recalibrate baseline priors.
2. **Single-Subject Focus:** Mathematical reasoning has unique failure patterns distinct from humanities or laboratory courses.
3. **Mid-Semester Grade Dependency:** High accuracy requires $G1$/$G2$ scores. The system provides an *Early-Semester Mode* toggle that forecasts risks without early grades.
4. **Proxy Approximations:** Continuous LMS interaction logs (Canvas/Blackboard telemetry) should be integrated in production environments.

---

## 🛠️ Built With

* **[Python 3.10+](https://www.python.org/)** — Core backend programming language
* **[Streamlit](https://streamlit.io/)** — Reactive web dashboard framework with live state caching
* **[Scikit-Learn](https://scikit-learn.org/)** — Classifiers, preprocessing pipelines, and metric evaluation
* **[Plotly Express & Graph Objects](https://plotly.com/)** — Interactive, responsive data visualizations
* **[Pandas & NumPy](https://pandas.pydata.org/)** — High-performance vector operations and cohort filtering

---

## 📜 License
This project is open-source under the **MIT License**.
