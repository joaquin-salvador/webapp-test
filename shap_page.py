"""
SHAP Explanations page — global importance, individual waterfall, and feature effects.
All plots show ALL features (no max_display cap).
"""

import streamlit as st
import pandas as pd
import numpy as np

from model import (
    _import_shap, _import_matplotlib, get_display_name, format_feature_value,
    build_person_label,
)


# ── Template Explanations ───────────────────────────────────────────────────

def explain_shap_waterfall(sample_values, shap_vals, features, display_names,
                           category_labels, likert_features, class_names, pred_class):
    feat_shap = list(zip(features, shap_vals, [sample_values[f] for f in features]))
    feat_shap.sort(key=lambda x: abs(x[1]), reverse=True)

    top_toward = [(f, s, v) for f, s, v in feat_shap if s > 0][:3]
    top_against = [(f, s, v) for f, s, v in feat_shap if s < 0][:3]

    lines = [f"**Why was this person classified as {class_names[pred_class]}?**", ""]

    if top_toward:
        lines.append("The main factors **pushing toward High Loneliness** are:")
        for feat, shap_val, val in top_toward:
            dname = get_display_name(feat, display_names)
            fval = format_feature_value(feat, val, category_labels, likert_features)
            lines.append(f"- **{dname}** = {fval} (contribution: +{shap_val:.3f})")

    if top_against:
        lines.append("")
        lines.append("The main factors **pushing toward Low Loneliness** are:")
        for feat, shap_val, val in top_against:
            dname = get_display_name(feat, display_names)
            fval = format_feature_value(feat, val, category_labels, likert_features)
            lines.append(f"- **{dname}** = {fval} (contribution: {shap_val:.3f})")

    lines.extend(["",
        "Each factor's contribution shows how much it pushes the prediction "
        "away from the average. Larger values mean stronger influence on the outcome."
    ])
    return "\n".join(lines)


def explain_global_importance(shap_vals_class, features, display_names, class_name):
    mean_abs = np.abs(shap_vals_class).mean(axis=0)
    ranked = sorted(zip(features, mean_abs), key=lambda x: x[1], reverse=True)

    lines = [
        f"**What drives the model's {class_name} predictions?**", "",
        "The chart above shows how much each feature influences the model on average. "
        "The top factors are:", ""
    ]
    for feat, imp in ranked[:5]:
        dname = get_display_name(feat, display_names)
        lines.append(f"- **{dname}** (average impact: {imp:.4f})")

    lines.extend(["",
        "Features at the top of the chart have the strongest influence on whether "
        "someone is classified in this category. This doesn't tell us the *direction* "
        "of the effect — see the Feature Effects tab for that."
    ])
    return "\n".join(lines)


# ── Page Renderer ───────────────────────────────────────────────────────────

def render_shap(model, X_explain, shap_values_test, shap_expected_value,
                features, class_names, display_names, category_labels,
                likert_features, precomputed_preds):
    shap = _import_shap()
    plt = _import_matplotlib()

    st.title("SHAP Explanations")

    st.info(
        "**SHAP (SHapley Additive exPlanations)** shows how each feature contributes "
        "to the model's prediction for each person. Think of it as a breakdown of "
        "*why* the model made a particular decision. "
        "SHAP values are computed via a **LightGBM surrogate** trained to mimic the "
        "primary TabPFN model, using exact TreeSHAP for fast and faithful explanations."
    )

    n_explained = len(X_explain)
    n_features = len(features)
    display_names_list = [get_display_name(f, display_names) for f in features]

    tab1, tab2, tab3 = st.tabs([
        "Global Feature Importance",
        "Individual Explanation",
        "Feature Effects"
    ])

    with tab1:
        st.subheader("Which features matter most overall?")
        st.markdown(
            "This chart shows the average importance of each feature across all "
            f"{n_explained} analyzed individuals. All {n_features} features are shown."
        )

        class_idx = st.selectbox("Show importance for:", range(len(class_names)),
                                 format_func=lambda x: class_names[x], key="shap_class")

        fig, ax = plt.subplots(figsize=(10, max(8, n_features * 0.35)))
        shap.summary_plot(shap_values_test[class_idx], X_explain,
                         feature_names=display_names_list, plot_type="bar",
                         show=False, max_display=n_features)
        plt.title(f"Feature Importance: {class_names[class_idx]}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("What does this mean?", expanded=True):
            st.markdown(explain_global_importance(
                shap_values_test[class_idx], features, display_names,
                class_names[class_idx]
            ))

    with tab2:
        st.subheader("Why was this specific person classified this way?")
        st.markdown(
            "Select a person from the test set to see which factors most influenced "
            "their individual prediction. Red bars push toward High Loneliness; "
            "blue bars push toward Low Loneliness."
        )

        person_options = list(range(n_explained))
        person_labels = {
            i: build_person_label(i, X_explain.iloc[i], class_names,
                                  display_names, category_labels, likert_features,
                                  precomputed_preds)
            for i in person_options
        }
        sample_idx = st.selectbox(
            "Select a person:", person_options,
            format_func=lambda x: person_labels[x],
            key="shap_person"
        )

        class_idx_ind = 1  # High Loneliness

        if precomputed_preds is not None and sample_idx < len(precomputed_preds["y_pred"]):
            pred = int(precomputed_preds["y_pred"][sample_idx])
            prob = np.array([1.0 - precomputed_preds["y_prob"][sample_idx],
                             precomputed_preds["y_prob"][sample_idx]])
        else:
            pred = model.predict(X_explain.iloc[[sample_idx]])[0]
            prob = model.predict_proba(X_explain.iloc[[sample_idx]])[0]

        col1, col2 = st.columns(2)
        col1.metric("Predicted Class", class_names[pred])
        col2.metric("Confidence", f"{prob[pred]:.1%} {class_names[pred]}")

        with st.expander("View this person's profile"):
            profile = []
            for feat in features:
                val = X_explain.iloc[sample_idx][feat]
                profile.append({
                    "Feature": get_display_name(feat, display_names),
                    "Value": format_feature_value(feat, val, category_labels, likert_features),
                    "Raw": f"{val:.2f}"
                })
            st.dataframe(pd.DataFrame(profile), use_container_width=True, hide_index=True)

        shap_explanation = shap.Explanation(
            values=shap_values_test[class_idx_ind][sample_idx],
            base_values=shap_expected_value[class_idx_ind]
                        if len(shap_expected_value) > 1
                        else float(shap_expected_value[0]),
            data=X_explain.iloc[sample_idx].values,
            feature_names=display_names_list
        )
        fig, ax = plt.subplots(figsize=(10, max(7, n_features * 0.3)))
        shap.waterfall_plot(shap_explanation, show=False, max_display=n_features)
        plt.title(f"What influenced this prediction? (Person {sample_idx})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("What does this mean?", expanded=True):
            st.markdown(explain_shap_waterfall(
                X_explain.iloc[sample_idx].to_dict(),
                shap_values_test[class_idx_ind][sample_idx],
                features, display_names, category_labels, likert_features,
                class_names, pred
            ))

    with tab3:
        st.subheader("How do features affect predictions?")
        st.markdown(
            "This beeswarm plot shows the relationship between each feature's "
            "value (color) and its impact on the prediction (position). "
            "**Red dots** = high feature values, **blue dots** = low values. "
            f"All {n_features} features are displayed."
        )

        class_idx_eff = st.selectbox("Class:", range(len(class_names)),
                                      format_func=lambda x: class_names[x],
                                      key="shap_eff_class")

        fig, ax = plt.subplots(figsize=(10, max(8, n_features * 0.35)))
        shap.summary_plot(shap_values_test[class_idx_eff], X_explain,
                         feature_names=display_names_list, show=False,
                         max_display=n_features)
        plt.title(f"Feature Effects: {class_names[class_idx_eff]}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("How to read this chart"):
            st.markdown(
                "- Each dot represents one person\n"
                "- **Horizontal position**: how much the feature pushed the prediction "
                "(right = toward this class, left = away)\n"
                "- **Color**: the actual value of the feature for that person "
                "(red = high, blue = low)\n"
                "- If red dots cluster on the right, high values of that feature "
                "push toward this class\n"
                "- Features are sorted by importance (most important at top)"
            )
