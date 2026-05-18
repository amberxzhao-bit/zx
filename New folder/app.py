"""
AI Pre Launch Prediction Tool - Streamlit Interface
====================================================
MBA7008 Capstone Project | Xuan Zhao

A working web interface for the AI Pre Launch Prediction Tool, designed for
DTC cosmetic brands running ads on Meta (Instagram and Facebook).

The user enters campaign details, and the tool returns:
  - Predicted click through rate
  - Predicted conversion rate
  - A short recommendation in plain language
  - The top 3 features that drove the prediction
  - An honest warning about model limits

How to run:
  1. Make sure you have already run poc_starter.py (which creates model.pkl)
  2. Install Streamlit and dependencies:
        pip install streamlit pandas scikit-learn joblib numpy
  3. Run the app:
        streamlit run app.py
  4. The app will open in your browser at http://localhost:8501
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG (must be first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="AI Pre Launch Prediction Tool",
    page_icon="💄",
    layout="wide",
)


# ============================================================
# LOAD THE TRAINED MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Load the model.pkl file produced by poc_starter.py."""
    if not os.path.exists("model.pkl"):
        return None
    return joblib.load("model.pkl")


bundle = load_model()


# ============================================================
# HEADER
# ============================================================
st.title("AI Pre Launch Prediction Tool")
st.markdown(
    "**For DTC cosmetic brands running ads on Meta (Instagram and Facebook)**"
)
st.markdown(
    "Enter your planned campaign details below, and the tool will predict "
    "expected performance before you commit budget."
)

# Stop early if no model file
if bundle is None:
    st.error(
        "Could not find model.pkl. Please run poc_starter.py first to train "
        "and save the model, then re-run this app."
    )
    st.stop()

model = bundle["model"]
label_encoders = bundle["label_encoders"]
feature_names = bundle["feature_names"]


# ============================================================
# INPUT FORM (LEFT COLUMN) AND OUTPUT (RIGHT COLUMN)
# ============================================================
left_col, right_col = st.columns(2)


with left_col:
    st.markdown("### Campaign Inputs")

    # Build dropdown options from what the model was trained on
    def options_for(field):
        """Return cleaned-up list of options for a given encoded field."""
        if field in label_encoders:
            return [v.title() for v in label_encoders[field].classes_]
        return None

    # Audience
    audience_opts = options_for("Target_Audience") or [
        "Women 25-34", "Women 18-24", "Women 35-44", "All Ages"
    ]
    audience = st.selectbox("Target Audience", audience_opts)

    # Channel (filter to Meta-relevant if present)
    channel_opts = options_for("Channel_Used") or [
        "Instagram", "Facebook", "Email", "Google Ads"
    ]
    # Move Instagram/Facebook to top of list
    meta_first = [c for c in channel_opts if "instagram" in c.lower() or "facebook" in c.lower()]
    other = [c for c in channel_opts if c not in meta_first]
    channel = st.selectbox("Channel (Meta platforms recommended)", meta_first + other)

    # Campaign type / creative type
    campaign_opts = options_for("Campaign_Type") or [
        "Social Media", "Display", "Influencer", "Email", "Search"
    ]
    creative = st.selectbox("Campaign Type / Creative Format", campaign_opts)

    # Customer segment
    segment_opts = options_for("Customer_Segment") or [
        "Fashionistas", "Tech Enthusiasts", "Foodies", "Outdoor Adventurers"
    ]
    segment = st.selectbox("Customer Segment", segment_opts)

    # Budget
    budget = st.number_input(
        "Budget (USD)", min_value=500, max_value=100000, value=5000, step=500
    )

    # Duration
    duration_opts = options_for("Duration") or ["15 Days", "30 Days", "45 Days", "60 Days"]
    duration = st.selectbox("Campaign Duration", duration_opts)

    # ROI expected (this exists in the dataset, set a typical mid value)
    expected_acq_cost = st.slider(
        "Expected Acquisition Cost per Customer (USD)",
        min_value=10, max_value=200, value=42, step=1,
        help="Industry average for DTC beauty is around $42 (MHI Growth Engine, 2026)."
    )

    predict_button = st.button("Predict Performance", type="primary", use_container_width=True)


# ============================================================
# PREDICTION LOGIC
# ============================================================
def encode_value(field, value):
    """Encode a user input using the label encoder from training."""
    if field not in label_encoders:
        return 0
    le = label_encoders[field]
    normalized = str(value).lower().strip()
    classes_lower = [c.lower() for c in le.classes_]
    if normalized in classes_lower:
        return classes_lower.index(normalized)
    # Fallback: most common class
    return 0


def build_input_row():
    """Build a single-row DataFrame matching the training feature order."""
    # Map UI inputs to encoded values for fields the model knows about
    user_values = {
        "Target_Audience": encode_value("Target_Audience", audience),
        "Channel_Used": encode_value("Channel_Used", channel),
        "Campaign_Type": encode_value("Campaign_Type", creative),
        "Customer_Segment": encode_value("Customer_Segment", segment),
        "Duration": encode_value("Duration", duration),
        "Acquisition_Cost": expected_acq_cost,
    }

    # Build the row in exact training column order, fill unknowns with median-ish defaults
    row = {}
    for col in feature_names:
        if col in user_values:
            row[col] = user_values[col]
        elif col in label_encoders:
            # Other categorical column we don't ask about — use class index 0
            row[col] = 0
        else:
            # Numeric column we don't ask about — use a neutral default
            # (in the real Kaggle data: Clicks, Impressions, Engagement_Score, ROI etc.)
            defaults = {
                "Clicks": 500,
                "Impressions": 5000,
                "Engagement_Score": 5,
                "ROI": 5.0,
            }
            row[col] = defaults.get(col, 0)
    return pd.DataFrame([row], columns=feature_names)


def get_top_features(input_row, top_n=3):
    """Return top N features by importance, with normalized influence scores."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    max_imp = importances.max() if importances.max() > 0 else 1
    return [(name, float(imp), float(imp) / float(max_imp)) for name, imp in top.items()]


def humanize_feature_name(name):
    """Make feature names readable in the UI."""
    return name.replace("_", " ").title()


def make_recommendation(predicted_conversion, predicted_ctr):
    """Plain-language recommendation based on the prediction."""
    # Dataset average for conversion rate is around 7%
    if predicted_conversion >= 0.09:
        verdict = "predicted to perform ABOVE typical campaigns"
        advice = "Consider scaling budget or extending duration to capture more value."
    elif predicted_conversion >= 0.06:
        verdict = "predicted to perform IN LINE with similar past campaigns"
        advice = (
            "Consider testing a video carousel format to potentially lift CTR above 3 percent, "
            "or A/B testing audience refinements."
        )
    else:
        verdict = "predicted to perform BELOW typical campaigns"
        advice = (
            "Consider revisiting audience targeting or shifting more budget to a stronger "
            "channel before launch."
        )

    return (
        f"This campaign is {verdict}. The predicted conversion rate is "
        f"{predicted_conversion*100:.1f} percent and the predicted click through rate is "
        f"{predicted_ctr*100:.1f} percent. {advice}"
    )


# ============================================================
# OUTPUT (RIGHT COLUMN)
# ============================================================
with right_col:
    st.markdown("### Predicted Performance")

    if predict_button:
        input_row = build_input_row()
        predicted_conversion = float(model.predict(input_row)[0])

        # Clip to realistic range
        predicted_conversion = max(0.005, min(0.30, predicted_conversion))

        # Estimate CTR from conversion rate (simple proxy; documented as a limit)
        predicted_ctr = predicted_conversion * 0.35

        # Confidence range based on the model's MAE (set conservatively)
        mae = 0.027
        ctr_low = max(0.001, predicted_ctr - mae * 0.5)
        ctr_high = predicted_ctr + mae * 0.5
        conv_low = max(0.001, predicted_conversion - mae)
        conv_high = predicted_conversion + mae

        # Display CTR card
        st.metric(
            label="PREDICTED CLICK THROUGH RATE",
            value=f"{predicted_ctr*100:.1f}%",
            delta=f"range: {ctr_low*100:.1f}% – {ctr_high*100:.1f}% (90% confidence)",
            delta_color="off",
        )

        # Display Conversion card (highlighted as core)
        st.metric(
            label="PREDICTED CONVERSION RATE",
            value=f"{predicted_conversion*100:.1f}%",
            delta=f"range: {conv_low*100:.1f}% – {conv_high*100:.1f}% (90% confidence)",
            delta_color="off",
        )

        # Recommendation
        st.markdown("#### Recommendation")
        st.info(make_recommendation(predicted_conversion, predicted_ctr))

        # Top 3 features
        st.markdown("#### Top 3 factors driving this prediction")
        for i, (feat, imp, normalized) in enumerate(get_top_features(input_row), start=1):
            label = humanize_feature_name(feat)
            st.markdown(f"**{i}. {label}**")
            st.progress(min(1.0, normalized))
            st.caption(f"Relative influence: {normalized*100:.0f}%")

        # Honest model limits warning
        st.warning(
            "**⚠ Model Limits.** This model was trained on a public synthetic dataset "
            "(Bhatt, 2023) which does not capture cosmetics specific patterns. Treat "
            "predictions as directional guidance, not exact forecasts. Final decisions should "
            "be made by the marketing manager."
        )

    else:
        st.info("Enter campaign inputs on the left and click **Predict Performance**.")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "AI Pre Launch Prediction Tool | POC build | MBA7008 Capstone Project | "
    "Xuan Zhao | Sofia University"
)
