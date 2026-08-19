# 🎓 Student Academic Performance Predictor & Early-Warning System (EWS)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning system developed to predict university student academic performance and identify **at-risk students ($G3 < 10$)** weeks prior to final examinations. Utilizing real-world academic, behavioral, and socioeconomic datasets (`student-mat.csv`), this project provides diagnostic visualizations, comparative model evaluations (Decision Tree vs. SVM), and actionable institutional intervention frameworks.

---

## 📌 Executive Summary

Academic failure adversely impacts student graduation rates, retention statistics, and institutional performance metrics. Traditional academic evaluations are **reactive**, assessing student status only after midterms or final exams when intervention is often too late.

This project implements a data-driven **Early-Warning System (EWS)** that:
1. **Ingests & Preprocesses** real-world student records ($N=395$).
2. **Conducts Exploratory Data Analysis (EDA)** to isolate key academic and social drivers of failure.
3. **Trains & Compares Machine Learning Classifiers** (`DecisionTreeClassifier` vs. `SVC`).
4. **Optimizes At-Risk Recall (69.23%)** to minimize False Negatives (unidentified struggling students).
5. **Establishes a 4-Tier University Support Roadmap** for early intervention.

---

## 📁 Repository Structure

```
PADV_ACT/
│
├── Education.ipynb                      
├── student-mat.csv                     
├── README.md                       

```

---

## 📊 Dataset Overview (`student-mat.csv`)

The dataset contains 395 student instances with 33 attributes covering academic history, demographic factors, social habits, and family background.

### Target Variable Engineering
$$\text{Target} = \begin{cases} 1 \quad (\text{Pass}), & \text{if } G3 \ge 10 \quad (265 \text{ students / } 67.09\%) \\ 0 \quad (\text{Fail / At-Risk}), & \text{if } G3 < 10 \quad (130 \text{ students / } 32.91\%) \end{cases}$$

### Key Feature Categories
* **Academic History**: `failures` (past class failures: 0-3), `studytime` (weekly study hours: 1-4), `G1`, `G2` (period grades).
* **Behavioral Habits**: `absences` (0-75 days), `goout` (1-5), `Dalc` (workday alcohol: 1-5), `Walc` (weekend alcohol: 1-5).
* **Socioeconomic Background**: `Medu` (mother's education: 0-4), `Fedu` (father's education: 0-4), `Mjob`, `Fjob`, `famrel` (family quality: 1-5).
* **Institutional Support**: `schoolsup` (educational support), `famsup` (family support), `internet` (home internet access).

---

## ⚙️ Machine Learning Pipeline Workflow

```
[ Data Ingestion ] ➔ [ Data Cleaning & Leakage Prevention ] ➔ [ LabelEncoding & Scaling ]
                                                                       │
[ University Interventions ] ⇇ [ Model Evaluation ] ⇇ [ Decision Tree / SVM Training ]
```

1. **Data Cleaning & Leakage Prevention**:
   * Executed missing value check (0 missing values across all features).
   * Excluded final exam grade $G3$ and intermediate status flags from predictor matrix $X$ to ensure real-world predictive validity before final exams.
2. **Encoding & Feature Scaling**:
   * Encoded categorical attributes using `LabelEncoder()`.
   * Standardized numerical features using `StandardScaler` ($z = \frac{x - \mu}{\sigma}$) for distance-sensitive models (SVM).
3. **Model Training**:
   * Split data into 80% Stratified Training ($N=316$) and 20% Hold-out Test ($N=79$) sets.
   * Trained `DecisionTreeClassifier(max_depth=4, criterion='gini')` and `SVC(kernel='rbf', C=1.0)`.

---

## 📈 Model Performance & Comparative Analysis

Models were evaluated across Accuracy, Precision, Recall for At-Risk Students, F1-Score, ROC-AUC, and 5-Fold Cross-Validation.

| Evaluation Metric | Decision Tree (depth=4) | SVM Classifier (RBF Kernel) | Selected Model |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **87.34%** | 79.75% | **Decision Tree (+7.59%)** |
| **Precision (Pass Class)** | **87.04%** | 79.69% | **Decision Tree (+7.35%)** |
| **Recall (Pass Class)** | 88.68% | **96.23%** | SVM |
| **Recall (Fail - At-Risk Class)** | **69.23%** | 42.31% | **Decision Tree (+26.92%)** |
| **F1-Score** | **0.9038** | 0.8718 | **Decision Tree (+0.0320)** |
| **ROC-AUC Score** | **0.8846** | 0.8766 | **Decision Tree (+0.0080)** |
| **5-Fold CV Accuracy** | **87.03%** | 79.75% | **Decision Tree (+7.28%)** |

### Why Decision Tree Was Selected
* **Higher At-Risk Recall (69.23%)**: Identifies **63.6% more failing students** than SVM ($42.31\%$), drastically reducing dangerous False Negatives.
* **Explainable Rules**: Generates explicit decision trees (e.g., `failures > 0.5 AND absences > 5.5`) that advisors can inspect and trust.
* **Generalization**: Achieves **87.03% 5-Fold Cross-Validation Accuracy** with no signs of overfitting.

---

## 🔍 High-Risk Student Profiles Identified

1. **Profile 1: Chronic Academic Backlog (Critical Risk)**
   * **Attributes**: Past failures $\ge 1$, weekly study time $< 2$ hours.
   * **Driver**: Accumulated learning deficits from previous terms compound as course difficulty increases.
2. **Profile 2: Attendance-Deficit & Disengaged (High Behavioral Risk)**
   * **Attributes**: Absences $> 10$ days, elevated alcohol consumption (`Walc`, `Dalc` $\ge 3$).
   * **Driver**: Chronic absenteeism leads to missed lectures and lower exam readiness.
3. **Profile 3: Socioeconomically Vulnerable (Moderate-High Risk)**
   * **Attributes**: Lower parental education (`Medu`, `Fedu` $\le 1$), lack of home internet (`internet = 0`).
   * **Driver**: Structural barriers and restricted home study resources.

---

## 🏛️ University Intervention Framework

1. **Week 4 Automated Screening**: Run the Decision Tree model at Week 4 of the semester to flag vulnerable students before midterms.
2. **Mandatory Peer Tutoring**: Auto-enroll students with past failures into peer-led study groups and remedial workshops.
3. **Absence Threshold Alerts**: Trigger automated advisor notifications whenever student absences exceed 6 days.
4. **Digital Equity Support**: Provide subsidized laptops and mobile internet access for socioeconomically disadvantaged students.

---

## 🚀 Quick Start & Installation

### Prerequisites
* Python 3.10 or higher
* Jupyter Notebook or VS Code Jupyter extension

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/student-academic-performance-ews.git
cd student-academic-performance-ews

python -m venv .venv
source .venv/bin/activate  
```

### 2. Install Required Packages
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook Education.ipynb
```

---

## 🛠️ Built With

* **[Python 3.10+](https://www.python.org/)** — Core programming language.
* **[Pandas](https://pandas.pydata.org/)** — Data manipulation and feature engineering.
* **[NumPy](https://numpy.org/)** — Numerical computation.
* **[Scikit-Learn](https://scikit-learn.org/)** — Machine learning algorithms, scaling, and metrics.
* **[Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)** — Visual diagnostics, confusion matrices, and ROC curves.

---

## 📜 License

This project is open-source under the **MIT License**. Feel free to adapt and build upon this work for academic research and institutional applications.
