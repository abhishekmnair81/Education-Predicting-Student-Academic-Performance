import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import streamlit as st

NOTEBOOK_BENCHMARK = {
    "Decision Tree": {
        "Accuracy": 0.8734,
        "Precision (Pass)": 0.9388,
        "Recall (Pass)": 0.8679,
        "Recall (Fail - Risk)": 0.8846,
        "F1-Score": 0.9020,
        "ROC-AUC": 0.9267,
        "5-Fold CV Accuracy": 0.8955,
    },
    "SVM": {
        "Accuracy": 0.7975,
        "Precision (Pass)": 0.8491,
        "Recall (Pass)": 0.8491,
        "Recall (Fail - Risk)": 0.6923,
        "F1-Score": 0.8491,
        "ROC-AUC": 0.8919,
        "5-Fold CV Accuracy": 0.8417,
    }
}

@st.cache_resource(show_spinner=False)
def train_and_evaluate_models(X_encoded_dict, y_series, include_g1_g2=True, tune_mode=False):
    """
    Trains Decision Tree and SVM models with Stratified 80/20 split (random_state=42).
    X_encoded_dict: dict or DataFrame of encoded features.
    """
    if isinstance(X_encoded_dict, dict):
        X = pd.DataFrame(X_encoded_dict)
    else:
        X = X_encoded_dict.copy()

    y = y_series.copy()
    feature_names = list(X.columns)

    # Stratified 80/20 Train-Test Split (316 train, 79 test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize numerical features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_params_dt = {}
    best_params_svm = {}

    if not tune_mode:
        # Default Models
        dt_model = DecisionTreeClassifier(
            max_depth=4,
            criterion="gini",
            min_samples_split=10,
            random_state=42
        )
        dt_model.fit(X_train, y_train)

        svm_model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        )
        svm_model.fit(X_train_scaled, y_train)
    else:
        # Tuned Models via 5-Fold GridSearchCV
        param_grid_dt = {
            "max_depth": [3, 4, 5, 6],
            "min_samples_split": [5, 10, 15],
            "criterion": ["gini", "entropy"]
        }
        grid_dt = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid=param_grid_dt,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        grid_dt.fit(X_train, y_train)
        dt_model = grid_dt.best_estimator_
        best_params_dt = grid_dt.best_params_

        param_grid_svm = {
            "C": [0.5, 1.0, 2.0, 5.0],
            "gamma": ["scale", 0.01, 0.1]
        }
        grid_svm = GridSearchCV(
            SVC(kernel="rbf", probability=True, random_state=42),
            param_grid=param_grid_svm,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        grid_svm.fit(X_train_scaled, y_train)
        svm_model = grid_svm.best_estimator_
        best_params_svm = grid_svm.best_params_

    # Predictions & Probabilities
    y_pred_dt = dt_model.predict(X_test)
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

    y_pred_svm = svm_model.predict(X_test_scaled)
    y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

    # Cross-validation on training data
    cv_dt = cross_val_score(dt_model, X_train, y_train, cv=5, scoring="accuracy").mean()
    cv_svm = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring="accuracy").mean()

    # Metrics computation helper
    def calc_metrics(y_true, y_pred, y_prob, cv_score):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (Pass)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "Recall (Pass)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "Recall (Fail - Risk)": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_prob),
            "5-Fold CV Accuracy": float(cv_score)
        }

    metrics_dt = calc_metrics(y_test, y_pred_dt, y_prob_dt, cv_dt)
    metrics_svm = calc_metrics(y_test, y_pred_svm, y_prob_svm, cv_svm)

    # Classification Reports
    report_dt = classification_report(y_test, y_pred_dt, target_names=["Fail (At-Risk)", "Pass"], digits=4)
    report_svm = classification_report(y_test, y_pred_svm, target_names=["Fail (At-Risk)", "Pass"], digits=4)

    # Confusion Matrices
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    cm_svm = confusion_matrix(y_test, y_pred_svm)

    # ROC Curves
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)

    # Feature Importances for DT
    feat_imp = pd.Series(dt_model.feature_importances_, index=feature_names).sort_values(ascending=False)

    # Decision tree rules text
    tree_rules_text = export_text(dt_model, feature_names=feature_names)

    return {
        "dt_model": dt_model,
        "svm_model": svm_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "metrics_dt": metrics_dt,
        "metrics_svm": metrics_svm,
        "report_dt": report_dt,
        "report_svm": report_svm,
        "cm_dt": cm_dt,
        "cm_svm": cm_svm,
        "roc_dt": (fpr_dt, tpr_dt, metrics_dt["ROC-AUC"]),
        "roc_svm": (fpr_svm, tpr_svm, metrics_svm["ROC-AUC"]),
        "feat_imp": feat_imp,
        "tree_rules_text": tree_rules_text,
        "best_params_dt": best_params_dt,
        "best_params_svm": best_params_svm,
        "tune_mode": tune_mode,
        "include_g1_g2": include_g1_g2
    }

def generate_tree_plot(dt_model, feature_names):
    """
    Renders the matplotlib plot_tree figure for Streamlit with luxury crisp styling.
    """
    fig, ax = plt.subplots(figsize=(19, 9.5), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=["Fail (At-Risk)", "Pass"],
        filled=True,
        rounded=True,
        fontsize=8.5,
        ax=ax
    )
    plt.title("Decision Tree Architectural Hierarchy & Risk Split Rules", fontsize=14, fontweight="bold", pad=14, color="#0F172A", family="sans-serif")
    plt.tight_layout()
    return fig

def is_matching_benchmark(metric_name, calculated_val, model_name="Decision Tree", tolerance=0.008):
    """
    Returns True if calculated value is within tolerance of reference notebook benchmark.
    """
    bench = NOTEBOOK_BENCHMARK.get(model_name, {}).get(metric_name)
    if bench is None:
        return False
    return abs(calculated_val - bench) <= tolerance
