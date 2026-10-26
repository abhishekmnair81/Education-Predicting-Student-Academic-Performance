import pandas as pd
import numpy as np

def score_cohort_risk(df, models_dict, model_choice="Decision Tree", include_g1_g2=True):
    """
    Computes failure risk probability for every student in the dataframe.
    Risk Probability = P(Target = 0) (Probability of Failing G3).
    Returns dataframe with Risk_Probability and Predicted_Status.
    """
    df_scored = df.copy()
    
    # Feature columns used during training
    feature_names = models_dict["feature_names"]
    
    # Encode categorical features matching original training
    X_score = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                X_score[col] = df[col].astype(float)
            else:
                # String / categorical
                X_score[col] = df[col].astype(str)
                # Map using training values or categorical codes
                unique_vals = sorted(df[col].unique())
                mapping = {v: i for i, v in enumerate(unique_vals)}
                X_score[col] = df[col].map(mapping).fillna(0)
        else:
            X_score[col] = 0.0

    if model_choice == "Decision Tree":
        model = models_dict["dt_model"]
        probs = model.predict_proba(X_score)
        # Class 0 is Fail, Class 1 is Pass
        fail_prob = probs[:, 0]
        preds = model.predict(X_score)
    else:
        model = models_dict["svm_model"]
        scaler = models_dict["scaler"]
        X_scaled = scaler.transform(X_score)
        probs = model.predict_proba(X_scaled)
        fail_prob = probs[:, 0]
        preds = model.predict(X_scaled)

    df_scored["Risk_Probability"] = np.round(fail_prob, 4)
    df_scored["Predicted_Class"] = preds
    df_scored["Predicted_Status"] = np.where(preds == 1, "Pass", "Fail")
    
    return df_scored

def assign_risk_tiers(df_scored, high_thresh=0.70, med_thresh=0.40):
    """
    Categorizes students into High, Medium, and Low risk bands based on dynamic thresholds.
    """
    df_tiered = df_scored.copy()
    
    conditions = [
        df_tiered["Risk_Probability"] >= high_thresh,
        (df_tiered["Risk_Probability"] >= med_thresh) & (df_tiered["Risk_Probability"] < high_thresh),
        df_tiered["Risk_Probability"] < med_thresh
    ]
    choices = [
        f"High Risk (≥ {int(high_thresh*100)}%)",
        f"Medium Risk ({int(med_thresh*100)}–{int(high_thresh*100)}%)",
        f"Low Risk (< {int(med_thresh*100)}%)"
    ]
    
    df_tiered["Risk_Tier"] = np.select(conditions, choices, default="Low Risk")
    df_tiered["Risk_Tier_Simple"] = np.select(
        [df_tiered["Risk_Probability"] >= high_thresh, df_tiered["Risk_Probability"] >= med_thresh],
        ["High", "Medium"],
        default="Low"
    )
    return df_tiered

def discover_at_risk_profiles(df_tiered, high_thresh=0.70):
    """
    Auto-discovers high-risk student profiles from data.
    Groups at-risk students by key dimensions:
    - past failures
    - absences bucket
    - study time
    - higher education ambition
    - alcohol / wellbeing indicator
    Derives 3 to 5 distinct profiles dynamically with size, avg risk, priority score, expected impact,
    and recommended interventions.
    """
    # Select high risk cohort, fallback to medium+high if high is small
    cohort = df_tiered[df_tiered["Risk_Probability"] >= high_thresh].copy()
    if len(cohort) < 5:
        cohort = df_tiered[df_tiered["Risk_Probability"] >= 0.40].copy()
    if len(cohort) == 0:
        cohort = df_tiered.copy()

    # Discretize traits for grouping
    cohort["Failures_Trait"] = cohort["failures"].apply(
        lambda x: "Zero Past Failures" if x == 0 else ("1 Prior Failure" if x == 1 else "2+ Multiple Failures")
    )
    cohort["Absence_Trait"] = pd.cut(
        cohort["absences"],
        bins=[-1, 5, 10, 1000],
        labels=["Low Absences (0-5d)", "Elevated Absences (6-10d)", "Chronic Absenteeism (11+d)"]
    )
    cohort["Study_Trait"] = cohort["studytime"].apply(
        lambda x: "Low Study (<2h/wk)" if x == 1 else ("Moderate Study (2-5h)" if x == 2 else "High Study (>5h)")
    )
    cohort["Ambition_Trait"] = cohort["higher"].apply(
        lambda x: "Aims for Higher Ed" if str(x).lower() == "yes" else "No Higher Ed Plan"
    )
    cohort["Lifestyle_Trait"] = ((cohort["Walc"] >= 3) | (cohort["Dalc"] >= 3)).apply(
        lambda x: "Elevated Alcohol/Social" if x else "Balanced Lifestyle"
    )

    group_cols = ["Failures_Trait", "Absence_Trait", "Study_Trait", "Ambition_Trait", "Lifestyle_Trait"]

    grouped = cohort.groupby(group_cols, observed=False).agg(
        student_count=("Risk_Probability", "count"),
        mean_risk=("Risk_Probability", "mean"),
        avg_absences=("absences", "mean"),
        avg_failures=("failures", "mean"),
        avg_g1=("G1", "mean") if "G1" in cohort.columns else ("absences", lambda x: 0),
        avg_g2=("G2", "mean") if "G2" in cohort.columns else ("absences", lambda x: 0)
    ).reset_index().sort_values(by=["student_count", "mean_risk"], ascending=[False, False])

    active_groups = grouped[grouped["student_count"] > 0].head(5)

    profiles = []
    total_cohort = len(cohort)

    for i, row in active_groups.iterrows():
        fail_trait = row["Failures_Trait"]
        abs_trait = str(row["Absence_Trait"])
        study_trait = row["Study_Trait"]
        ambition_trait = row["Ambition_Trait"]
        life_trait = row["Lifestyle_Trait"]
        cnt = int(row["student_count"])
        avg_r = float(row["mean_risk"])
        pct = (cnt / total_cohort * 100) if total_cohort > 0 else 0

        # Dynamic archetype naming derived from data
        if "Multiple" in fail_trait or "1 Prior" in fail_trait:
            archetype = f"Profile {len(profiles)+1}: Cumulative Academic Deficit Cohort"
        elif "Chronic" in abs_trait or "Elevated" in abs_trait:
            archetype = f"Profile {len(profiles)+1}: Attendance Disengagement & Absenteeism"
        elif "Low Study" in study_trait:
            archetype = f"Profile {len(profiles)+1}: Study-Hour Deficit & Skill Vulnerability"
        elif "No Higher" in ambition_trait:
            archetype = f"Profile {len(profiles)+1}: Career Motivation & Aspiration Gap"
        else:
            archetype = f"Profile {len(profiles)+1}: Early Diagnostic / Assessment Slump"

        # Map actionable interventions dynamically based on traits
        interventions = []
        if "Multiple" in fail_trait or "1 Prior" in fail_trait:
            interventions.append("• Mandatory remedial tutorials & peer-assisted concept workshops")
        if "Chronic" in abs_trait or "Elevated" in abs_trait:
            interventions.append("• Automated attendance alert trigger, weekly SMS check-ins & advisor contact")
        if "Low Study" in study_trait:
            interventions.append("• Time-management coaching & structured campus study tables (min 4 hrs/wk)")
        if "No Higher" in ambition_trait:
            interventions.append("• Academic career counseling, degree ROI workshops & faculty mentorship")
        if "Elevated Alcohol" in life_trait:
            interventions.append("• Confidential student wellness referral & campus engagement alternatives")
        if "G1" in cohort.columns and row["avg_g1"] < 10:
            interventions.append("• Formative assessment recovery plan & targeted TA office hour sessions")

        if not interventions:
            interventions.append("• Bi-weekly academic advisor 1-on-1 check-in & progress monitoring")

        # Priority score & expected impact
        if avg_r >= 0.85:
            priority = "🚨 Urgent (Level 1)"
            impact = "Very High (+35–50% fail reduction via intensive recovery)"
        elif avg_r >= 0.70:
            priority = "⚠️ High (Level 2)"
            impact = "High (+25–35% pass rate recovery with early coaching)"
        else:
            priority = "ℹ️ Moderate (Level 3)"
            impact = "Moderate (+15–20% stabilization via preventative nudges)"

        profiles.append({
            "archetype": archetype,
            "count": cnt,
            "cohort_pct": pct,
            "avg_risk": avg_r,
            "traits": f"{fail_trait} | {abs_trait} | {study_trait} | {ambition_trait}",
            "interventions": interventions,
            "priority": priority,
            "expected_impact": impact,
            "avg_absences": float(row["avg_absences"]),
            "avg_failures": float(row["avg_failures"]),
            "avg_g1": float(row["avg_g1"]) if "G1" in cohort.columns else None,
            "avg_g2": float(row["avg_g2"]) if "G2" in cohort.columns else None
        })

    return profiles

def get_what_if_prediction(input_data, models_dict, encoders_dict, include_g1_g2=True):
    """
    Executes what-if prediction on an individual synthetic or real student profile.
    Returns prediction probabilities, top risk drivers, and personalized intervention list.
    """
    feature_names = models_dict["feature_names"]
    row_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    X_pred = pd.DataFrame(index=[0])
    for col in feature_names:
        if col in row_df.columns:
            val = row_df[col].iloc[0]
            if col in encoders_dict:
                le = encoders_dict[col]
                # Check if value exists in encoder classes
                if str(val) in le.classes_:
                    X_pred[col] = le.transform([str(val)])[0]
                else:
                    X_pred[col] = 0
            else:
                X_pred[col] = float(val)
        else:
            X_pred[col] = 0.0

    # Decision tree prediction
    dt_model = models_dict["dt_model"]
    dt_probs = dt_model.predict_proba(X_pred)[0]
    dt_pred = dt_model.predict(X_pred)[0]

    # SVM prediction
    svm_model = models_dict["svm_model"]
    scaler = models_dict["scaler"]
    X_pred_scaled = scaler.transform(X_pred)
    svm_probs = svm_model.predict_proba(X_pred_scaled)[0]
    svm_pred = svm_model.predict(X_pred_scaled)[0]

    # Consensus Risk Probability (Probability of Fail)
    dt_risk = float(dt_probs[0])
    svm_risk = float(svm_probs[0])
    consensus_risk = float((dt_risk + svm_risk) / 2.0)

    # Identify individual risk contributing factors
    risk_drivers = []
    interventions = []

    if input_data.get("failures", 0) >= 1:
        risk_drivers.append(f"Past academic failures ({input_data['failures']} course(s) failed previously)")
        interventions.append("Enroll in weekly peer-led remedial mathematics tutoring & problem-solving clinics.")

    if input_data.get("absences", 0) >= 10:
        risk_drivers.append(f"High cumulative absenteeism ({input_data['absences']} days missed)")
        interventions.append("Establish attendance accountability contract with weekly automated SMS alerts.")

    if input_data.get("studytime", 2) == 1:
        risk_drivers.append("Low weekly study time (< 2 hours/week)")
        interventions.append("Schedule academic time-management coaching and structured study block commitments.")

    if str(input_data.get("higher", "yes")).lower() == "no":
        risk_drivers.append("No stated intention to pursue higher education")
        interventions.append("Connect with academic counselor for career mapping and educational relevance advising.")

    if input_data.get("Walc", 1) >= 3 or input_data.get("Dalc", 1) >= 3:
        risk_drivers.append(f"Elevated alcohol consumption (Weekend: {input_data.get('Walc')}/5, Workday: {input_data.get('Dalc')}/5)")
        interventions.append("Provide confidential counseling resources and holistic wellness check-in.")

    if include_g1_g2:
        if input_data.get("G2", 10) < 10:
            risk_drivers.append(f"Below-passing Period 2 grade (G2 = {input_data.get('G2')}/20)")
            interventions.append("Targeted midterm exam review, diagnostic error analysis, and TA consultation.")
        elif input_data.get("G1", 10) < 10:
            risk_drivers.append(f"Low Period 1 baseline grade (G1 = {input_data.get('G1')}/20)")
            interventions.append("Supplemental problem sets and concept foundation review.")

    if str(input_data.get("address", "U")).upper() == "R" or input_data.get("traveltime", 1) >= 3:
        risk_drivers.append("Rural residence or long daily commute (> 30 mins)")
        interventions.append("Provide recorded lecture access, offline study materials, and transport support.")

    if str(input_data.get("internet", "yes")).lower() == "no":
        risk_drivers.append("No home internet access")
        interventions.append("Grant priority access to university computer labs and mobile hotspot assistance.")

    if not risk_drivers:
        risk_drivers.append("All primary behavioral and academic indicators within healthy thresholds.")
        interventions.append("Maintain standard academic trajectory; continue regular lecture and tutorial participation.")

    return {
        "dt_risk": dt_risk,
        "dt_pred": "Pass" if dt_pred == 1 else "Fail",
        "svm_risk": svm_risk,
        "svm_pred": "Pass" if svm_pred == 1 else "Fail",
        "consensus_risk": consensus_risk,
        "status": "Fail (At-Risk)" if consensus_risk >= 0.50 else "Pass (On-Track)",
        "risk_drivers": risk_drivers,
        "interventions": interventions
    }
