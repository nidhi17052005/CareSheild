"""
FWA Claim Risk Scorer — Streamlit UI
Uses the same cleaning, feature engineering, rule scoring, Isolation Forest,
and risk-tier logic as FWA_Datathon.ipynb.

How to execute
--------------
1. Open a terminal in this folder (Healthcare).
2. Create / activate a virtual environment (optional but recommended):
     python -m venv .venv
     .venv\\Scripts\\activate          # Windows
     source .venv/bin/activate        # macOS / Linux
3. Install dependencies:
     pip install streamlit pandas numpy scikit-learn openpyxl
4. Run the app:
     streamlit run app.py
5. In the browser, upload a claims Excel (.xlsx / .xls) or CSV file
   with the same columns as 01_Healthcare_CareShield_RAW.xlsx.
"""

from __future__ import annotations

import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = [
    "claim_id",
    "patient_age",
    "gender",
    "provider",
    "diagnosis",
    "procedure",
    "length_of_stay",
    "claimed_amount",
    "approved_amount",
    "tests_count",
    "visits_last_30_days",
    "duplicate_bill_flag",
    "claim_channel",
]

NUMERIC_COLUMNS = [
    "patient_age",
    "length_of_stay",
    "claimed_amount",
    "approved_amount",
    "tests_count",
    "visits_last_30_days",
]

OUTPUT_COLUMNS = [
    "claim_id",
    "provider_clean",
    "procedure",
    "claimed_amount",
    "approved_amount",
    "risk_score",
    "risk_tier",
    "rule_score",
    "ml_score",
    "flag_reasons",
]

GENDER_MAP = {"m": "M", "male": "M", "f": "F", "female": "F"}

SUSPICIOUS_TIERS = ["Highly Suspicious", "Unusual"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for required in REQUIRED_COLUMNS:
        if required in df.columns:
            continue
        if required.lower() in lower_map:
            rename[lower_map[required.lower()]] = required
    return df.rename(columns=rename)


def validate_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def load_claims(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload a .csv, .xlsx, or .xls file.")


def score_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates FWA_Datathon.ipynb scoring end-to-end."""
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Data Cleaning & Standardization
    df["gender_clean"] = (
        df["gender"].astype(str).str.strip().str.lower().map(GENDER_MAP).fillna("Other")
    )
    df["provider_clean"] = df["provider"].astype(str).str.strip().str.title()
    df["claim_channel_clean"] = df["claim_channel"].astype(str).str.strip().str.title()
    df["duplicate_bill_flag_clean"] = (
        df["duplicate_bill_flag"].astype(str).str.strip().str.capitalize()
    )

    # 2. Feature Engineering
    claimed = df["claimed_amount"].replace(0, np.nan)
    df["discrepancy_ratio"] = (df["claimed_amount"] - df["approved_amount"]) / claimed
    df["cost_per_day"] = df["claimed_amount"] / df["length_of_stay"].replace(0, 1)
    df["test_to_visit_ratio"] = df["tests_count"] / (df["visits_last_30_days"] + 1)

    proc_medians = df.groupby("procedure")["claimed_amount"].transform("median")
    proc_stds = df.groupby("procedure")["claimed_amount"].transform("std").fillna(1.0).replace(0, 1.0)
    df["procedure_cost_zscore"] = (df["claimed_amount"] - proc_medians) / proc_stds

    # 3. Signal A: Rule-Based Flags & Scores
    rule_scores = []
    rule_reasons = []

    for _, row in df.iterrows():
        score = 0
        reasons = []

        if row["duplicate_bill_flag_clean"] == "Yes":
            score += 35
            reasons.append("Duplicate bill flag present")

        zscore = row["procedure_cost_zscore"]
        if pd.notna(zscore) and zscore > 1.5:
            score += 30
            reasons.append(
                f"Claim amount significantly higher than procedure median (+{zscore:.1f}σ)"
            )

        if row["length_of_stay"] == 0 and pd.notna(row["claimed_amount"]) and row["claimed_amount"] > 5000:
            score += 25
            reasons.append("High claim amount with 0 days length of stay")

        if (
            pd.notna(row["visits_last_30_days"])
            and pd.notna(row["tests_count"])
            and row["visits_last_30_days"] >= 4
            and row["tests_count"] >= 2
        ):
            score += 15
            reasons.append("High visit count (>3 in 30 days) with repeated tests")

        rule_scores.append(min(score, 100))
        rule_reasons.append("; ".join(reasons) if reasons else "No standard rule flags")

    df["rule_score"] = rule_scores
    df["flag_reasons"] = rule_reasons

    # Signal B: Isolation Forest (same features, contamination, random_state)
    features = [
        "length_of_stay",
        "claimed_amount",
        "discrepancy_ratio",
        "cost_per_day",
        "procedure_cost_zscore",
    ]
    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    if len(df) < 2:
        df["ml_score"] = 0.0
    else:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)
        contamination = 0.25 if len(df) >= 8 else min(0.25, max(0.05, 1 / len(df)))
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(x_scaled)
        anomaly_raw = -iso.score_samples(x_scaled)
        if np.allclose(anomaly_raw.min(), anomaly_raw.max()):
            df["ml_score"] = 0.0
        else:
            df["ml_score"] = np.interp(
                anomaly_raw, (anomaly_raw.min(), anomaly_raw.max()), (0, 100)
            )

    df["risk_score"] = (0.5 * df["rule_score"] + 0.5 * df["ml_score"]).round(1)

    conditions = [
        df["risk_score"] >= 65,
        (df["risk_score"] >= 35) & (df["risk_score"] < 65),
        df["risk_score"] < 35,
    ]
    df["risk_tier"] = np.select(
        conditions, ["Highly Suspicious", "Unusual", "Normal"], default="Unknown"
    )
    return df


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="risk_scores")
    return buffer.getvalue()


def render_summary(scored: pd.DataFrame) -> None:
    total = len(scored)
    highly = scored[scored["risk_tier"] == "Highly Suspicious"]
    unusual = scored[scored["risk_tier"] == "Unusual"]
    flagged = scored[scored["risk_tier"].isin(SUSPICIOUS_TIERS)]
    normal = scored[scored["risk_tier"] == "Normal"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims", f"{total:,}")
    c2.metric(
        "Highly suspicious",
        f"{len(highly):,}",
        f"{(len(highly) / total * 100) if total else 0:.1f}%",
    )
    c3.metric(
        "Suspicious (Unusual)",
        f"{len(unusual):,}",
        f"{(len(unusual) / total * 100) if total else 0:.1f}%",
    )
    c4.metric(
        "Normal",
        f"{len(normal):,}",
        f"{(len(normal) / total * 100) if total else 0:.1f}%",
    )

    st.subheader("Flagged claims summary")
    st.caption(
        "Highly Suspicious = risk_score ≥ 65. Suspicious / Unusual = 35 ≤ risk_score < 65. "
        "Composite score is 50% rule-based + 50% Isolation Forest anomaly score."
    )

    if flagged.empty:
        st.success("No suspicious or highly suspicious claims were found.")
        return

    st.markdown(
        f"**{len(flagged):,}** of **{total:,}** claims are flagged "
        f"(**{len(highly):,}** highly suspicious, **{len(unusual):,}** unusual)."
    )

    if not highly.empty:
        st.markdown("#### Highly Suspicious")
        st.dataframe(
            highly[OUTPUT_COLUMNS].sort_values("risk_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    if not unusual.empty:
        st.markdown("#### Suspicious (Unusual)")
        st.dataframe(
            unusual[OUTPUT_COLUMNS].sort_values("risk_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Flagged amount at risk")
    amount_cols = st.columns(2)
    highly_amt = highly["claimed_amount"].fillna(0).sum()
    unusual_amt = unusual["claimed_amount"].fillna(0).sum()
    amount_cols[0].metric("Highly suspicious claimed amount", f"{highly_amt:,.2f}")
    amount_cols[1].metric("Suspicious (unusual) claimed amount", f"{unusual_amt:,.2f}")

    by_provider = (
        flagged.groupby(["provider_clean", "risk_tier"], dropna=False)
        .agg(claims=("claim_id", "count"), claimed_amount=("claimed_amount", "sum"))
        .reset_index()
        .sort_values("claims", ascending=False)
    )
    st.markdown("#### Flagged claims by provider")
    st.dataframe(by_provider, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="FWA Claim Risk Scorer", layout="wide")
    st.title("FWA Claim Risk Scorer")
    st.write("Upload csv to get results")

    with st.sidebar:
        st.header("How to run")
        st.markdown(
            """
1. `pip install streamlit pandas numpy scikit-learn openpyxl`
2. `streamlit run app.py`
3. Upload `.xlsx` / `.xls` / `.csv`
            """
        )
        st.header("Required columns")
        st.code("\n".join(REQUIRED_COLUMNS), language="text")
        st.header("Risk tiers")
        st.markdown(
            """
- **Highly Suspicious:** risk_score ≥ 65
- **Unusual (suspicious):** 35–64.9
- **Normal:** risk_score < 35

**Rules (max 100):** duplicate bill +35; procedure cost z-score > 1.5 +30;
0-day stay with claim > 5000 +25; ≥4 visits in 30 days and ≥2 tests +15.

**ML:** Isolation Forest on length of stay, claimed amount, discrepancy ratio,
cost per day, and procedure cost z-score (`contamination=0.25`).
            """
        )

    uploaded = st.file_uploader(
        "Upload claims file",
        type=["csv", "xlsx", "xls"],
        help="Same schema as 01_Healthcare_CareShield_RAW.xlsx",
    )

    if uploaded is None:
        st.info("Upload a claims file to generate the suspicious / highly suspicious summary.")
        return

    try:
        raw = load_claims(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return

    raw = _normalize_columns(raw)
    missing = validate_columns(raw)
    if missing:
        st.error("The file is missing required columns: " + ", ".join(missing))
        st.write("Columns found:", list(raw.columns))
        return

    with st.expander("Preview uploaded data", expanded=False):
        st.dataframe(raw.head(20), use_container_width=True, hide_index=True)

    with st.spinner("Scoring claims with the FWA Datathon model…"):
        scored = score_claims(raw)

    render_summary(scored)

    st.subheader("Full scored output")
    show_all = st.checkbox("Show all claims (including Normal)", value=False)
    display = scored if show_all else scored[scored["risk_tier"].isin(SUSPICIOUS_TIERS)]
    if display.empty and not show_all:
        st.write("No flagged rows to display. Enable “Show all claims” to see Normal rows.")
    else:
        st.dataframe(
            display[OUTPUT_COLUMNS].sort_values(
                ["risk_score", "claim_id"], ascending=[False, True]
            ),
            use_container_width=True,
            hide_index=True,
        )

    csv_bytes = scored[OUTPUT_COLUMNS].to_csv(index=False).encode("utf-8")
    xlsx_bytes = dataframe_to_excel_bytes(scored[OUTPUT_COLUMNS])
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download scored CSV",
        data=csv_bytes,
        file_name="fwa_risk_scores.csv",
        mime="text/csv",
    )
    dl2.download_button(
        "Download scored Excel",
        data=xlsx_bytes,
        file_name="fwa_risk_scores.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
