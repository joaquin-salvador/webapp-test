"""
XAI Dashboard — UCLA Loneliness Classification
================================================
Streamlit web app providing:
1. Model Overview with performance metrics
2. SHAP-based global and individual explanations (via TreeSHAP surrogate)
3. What-If Analysis with interactive feature adjustment (surrogate predictions)
4. Counterfactual Explorer with pre-computed results
"""

import streamlit as st
from model import load_artifacts, DEFAULT_CATEGORY_LABELS, DEFAULT_DISPLAY_NAMES, DEFAULT_LIKERT_FEATURES

st.set_page_config(
    page_title="XAI Dashboard — UCLA Loneliness",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    try:
        (model, surrogate, X_train, X_test, X_explain, y_train, y_test,
         shap_values_test, shap_expected_value, feature_info,
         precomputed_preds, precomputed_cfs) = load_artifacts()
    except FileNotFoundError as e:
        st.error(
            "Could not load model artifacts. Please run the notebook first "
            "to generate them, then place the output files in the `data/` folder.\n\n"
            f"Error: {e}"
        )
        st.markdown(
            "**Expected folder structure:**\n"
            "```\n"
            "finalfinal_webapp/\n"
            "  data/\n"
            "    models/tabpfn.joblib\n"
            "    models/surrogate_lgbm.joblib\n"
            "    explainers/shap_values_test.pkl\n"
            "    explainers/shap_expected_value.pkl\n"
            "    explainers/feature_info.pkl\n"
            "    explainers/counterfactual_results.pkl\n"
            "    artifacts/X_train.csv\n"
            "    artifacts/X_test.csv\n"
            "    artifacts/y_train.csv\n"
            "    artifacts/y_test.csv\n"
            "    artifacts/X_explain.csv\n"
            "    artifacts/test_predictions.pkl\n"
            "```"
        )
        st.stop()

    features = feature_info["features"]
    class_names = feature_info["class_names"]
    category_labels = feature_info.get("category_labels", DEFAULT_CATEGORY_LABELS)
    display_names = feature_info.get("display_names", DEFAULT_DISPLAY_NAMES)
    likert_features = feature_info.get("likert_features", DEFAULT_LIKERT_FEATURES)

    # --- Sidebar ---
    st.sidebar.title("UCLA Loneliness XAI Dashboard")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate to:", [
        "Model Overview",
        "SHAP Explanations",
        "What-If Analysis",
        "Counterfactual Explorer",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About the Model**")
    st.sidebar.markdown(f"- Primary model: {type(model).__name__}")
    if surrogate is not None:
        st.sidebar.markdown(f"- Surrogate: {type(surrogate).__name__} (TreeSHAP + What-If)")
    st.sidebar.markdown(f"- Features: {len(features)}")
    st.sidebar.markdown(f"- Test samples: {len(X_test)}")
    st.sidebar.markdown(f"- SHAP explained: {len(X_explain)} samples")
    if precomputed_cfs:
        st.sidebar.markdown(f"- Counterfactuals: {len(precomputed_cfs)} samples")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**UCLA Loneliness Scale**")
    st.sidebar.markdown("Score < 22: Low Loneliness")
    st.sidebar.markdown("Score >= 22: High Loneliness")

    # --- Pages ---
    if page == "Model Overview":
        from model import render_overview
        render_overview(model, X_test, y_test, class_names, precomputed_preds)
    elif page == "SHAP Explanations":
        from shap_page import render_shap
        render_shap(model, X_explain, shap_values_test, shap_expected_value,
                    features, class_names, display_names, category_labels,
                    likert_features, precomputed_preds)
    elif page == "What-If Analysis":
        from counterfactual import render_whatif
        render_whatif(model, surrogate, X_train, X_explain, shap_values_test,
                      shap_expected_value, features, class_names,
                      display_names, category_labels, likert_features,
                      precomputed_preds)
    elif page == "Counterfactual Explorer":
        from counterfactual import render_counterfactuals
        render_counterfactuals(model, X_test, y_test, features, class_names,
                               feature_info, display_names, category_labels,
                               likert_features, precomputed_cfs, precomputed_preds)


if __name__ == "__main__":
    main()
