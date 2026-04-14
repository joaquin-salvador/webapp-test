"""
What-If Analysis and Counterfactual Explorer pages.
What-If uses the LightGBM surrogate model for fast predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np

from model import (
    _import_shap, _import_matplotlib, _import_plotly,
    get_display_name, format_feature_value, build_person_label,
    BINARY_FEATURES, JOB_FEATURES, JOB_OPTIONS,
)


# ── Template Explanations ───────────────────────────────────────────────────

def explain_whatif(original_values, modified_values, features, display_names,
                   category_labels, likert_features, shap_vals,
                   class_names, orig_pred, mod_pred, orig_prob, mod_prob):
    changed = []
    for feat in features:
        orig = float(original_values[feat])
        mod = float(modified_values[feat])
        if abs(mod - orig) > 1e-9:
            idx = features.index(feat)
            changed.append((feat, orig, mod, shap_vals[idx]))

    if not changed:
        return "No features were changed. Adjust the sliders above to see how the prediction changes."

    changed.sort(key=lambda x: abs(x[3]), reverse=True)
    pred_flipped = orig_pred != mod_pred
    lines = []

    if pred_flipped:
        lines.append(
            f"The prediction **changed** from **{class_names[orig_pred]}** "
            f"to **{class_names[mod_pred]}** "
            f"(probability shifted from {orig_prob[1]:.1%} to {mod_prob[1]:.1%})."
        )
    else:
        lines.append(
            f"The prediction **stayed** as **{class_names[orig_pred]}** "
            f"(probability shifted from {orig_prob[1]:.1%} to {mod_prob[1]:.1%})."
        )

    lines.extend(["", "**Changes made and their importance:**"])
    for feat, orig, mod, shap_val in changed[:5]:
        dname = get_display_name(feat, display_names)
        orig_label = format_feature_value(feat, orig, category_labels, likert_features)
        mod_label = format_feature_value(feat, mod, category_labels, likert_features)
        importance = "high" if abs(shap_val) > 0.05 else "moderate" if abs(shap_val) > 0.02 else "low"
        direction = "toward High Loneliness" if shap_val > 0 else "toward Low Loneliness"
        lines.append(
            f"- **{dname}**: {orig_label} -> {mod_label} "
            f"({importance} importance, originally pushed {direction})"
        )

    lines.append("")
    if pred_flipped:
        lines.append(
            "The combination of these changes was enough to tip the model's "
            "prediction. The features with highest importance had the most "
            "influence on the outcome."
        )
    else:
        lines.append(
            "These changes were not enough to flip the prediction. "
            "Try adjusting the features with higher importance values."
        )
    return "\n".join(lines)


def explain_counterfactual(original_values, cf_record, features, display_names,
                           category_labels, likert_features, class_names, original_pred):
    target_class = class_names[1 - original_pred]
    original_class = class_names[original_pred]

    changes = []
    for feat in features:
        if feat in cf_record and feat in original_values:
            orig = original_values[feat]
            cf_val = cf_record[feat]
            if abs(cf_val - orig) > 0.01:
                changes.append((feat, orig, cf_val))

    if not changes:
        return "No significant changes were needed in this counterfactual scenario."

    lines = [
        f"This person is currently classified as **{original_class}**. "
        f"To change the prediction to **{target_class}**, "
        f"the following changes would be needed:", ""
    ]
    for feat, orig, cf_val in changes:
        dname = get_display_name(feat, display_names)
        orig_label = format_feature_value(feat, orig, category_labels, likert_features)
        cf_label = format_feature_value(feat, cf_val, category_labels, likert_features)
        direction = "increase" if cf_val > orig else "decrease"
        lines.append(f"- **{dname}**: {orig_label} -> {cf_label} ({direction})")

    lines.extend(["",
        "These are hypothetical scenarios showing what the model considers "
        "important, not guaranteed real-world outcomes. Changes in one factor "
        "may interact with others in complex ways."
    ])
    return "\n".join(lines)


# ── What-If Page ────────────────────────────────────────────────────────────

def render_whatif(model, surrogate, X_train, X_explain, shap_values_test,
                  shap_expected_value, features, class_names,
                  display_names, category_labels, likert_features,
                  precomputed_preds):
    shap = _import_shap()
    plt = _import_matplotlib()

    # Use surrogate for modified predictions (much faster than TabPFN)
    predict_model = surrogate if surrogate is not None else model

    st.title("What-If Analysis")

    st.info(
        "Explore how changing a person's characteristics would affect the model's "
        "prediction. Select a person, adjust their features using the controls below, "
        "and see the result instantly."
        + (" Predictions use the **LightGBM surrogate** for fast response."
           if surrogate is not None else "")
    )

    n_explained = len(X_explain)

    person_options = list(range(n_explained))
    person_labels = {
        i: build_person_label(i, X_explain.iloc[i], class_names,
                              display_names, category_labels, likert_features,
                              precomputed_preds)
        for i in person_options
    }
    sample_idx = st.selectbox(
        "Start from person:", person_options,
        format_func=lambda x: person_labels[x],
        key="wi_sample"
    )

    original = X_explain.iloc[sample_idx]

    if precomputed_preds is not None and sample_idx < len(precomputed_preds["y_pred"]):
        orig_pred = int(precomputed_preds["y_pred"][sample_idx])
        orig_prob = np.array([1.0 - precomputed_preds["y_prob"][sample_idx],
                              precomputed_preds["y_prob"][sample_idx]])
    else:
        orig_pred = model.predict(X_explain.iloc[[sample_idx]])[0]
        orig_prob = model.predict_proba(X_explain.iloc[[sample_idx]])[0]

    st.markdown("---")
    st.subheader("Adjust Features")
    st.markdown("Use the controls below to change this person's characteristics "
                "and see how the prediction responds.")

    feat_min = X_train[features].min()
    feat_max = X_train[features].max()

    modified_values = {}
    job_handled = False
    col_idx = 0
    cols = st.columns(3)

    for feat in features:
        orig_val = float(original[feat])
        dname = get_display_name(feat, display_names)

        if feat in JOB_FEATURES:
            if not job_handled:
                job_handled = True
                col = cols[col_idx % 3]
                col_idx += 1

                orig_job = "Job_Employed"
                for jf in JOB_FEATURES:
                    if float(original[jf]) == 1.0:
                        orig_job = jf
                        break

                with col:
                    selected_job = st.selectbox(
                        "Occupation",
                        options=JOB_FEATURES,
                        format_func=lambda x: JOB_OPTIONS[x],
                        index=JOB_FEATURES.index(orig_job),
                        key="wi_occupation"
                    )

                for jf in JOB_FEATURES:
                    modified_values[jf] = 1.0 if jf == selected_job else 0.0
            continue

        col = cols[col_idx % 3]
        col_idx += 1

        with col:
            if feat in BINARY_FEATURES:
                if feat in category_labels:
                    options_map = category_labels[feat]
                    selected = st.selectbox(
                        dname, options=list(options_map.keys()),
                        format_func=lambda x, m=options_map: m[x],
                        index=int(orig_val), key=f"wi_{feat}"
                    )
                    modified_values[feat] = float(selected)
                else:
                    modified_values[feat] = float(
                        st.selectbox(dname, options=[0, 1],
                                     index=int(orig_val), key=f"wi_{feat}")
                    )
            elif feat == "Income":
                cats = category_labels.get("Income", {})
                selected = st.selectbox(
                    dname, options=list(cats.keys()),
                    format_func=lambda x, m=cats: m[x],
                    index=int(orig_val), key=f"wi_{feat}"
                )
                modified_values[feat] = float(selected)
            elif feat in likert_features:
                modified_values[feat] = float(
                    st.slider(f"{dname} (1-7 scale)",
                              min_value=1, max_value=7,
                              value=int(round(orig_val)), step=1,
                              key=f"wi_{feat}")
                )
            elif feat == "Age":
                modified_values[feat] = float(
                    st.slider(dname,
                              min_value=19, max_value=92,
                              value=int(round(orig_val)), step=1,
                              key=f"wi_{feat}")
                )
            else:
                lo = int(feat_min[feat])
                hi = int(feat_max[feat])
                lo = min(lo, int(round(orig_val)))
                hi = max(hi, int(round(orig_val)))
                modified_values[feat] = float(
                    st.slider(dname, min_value=lo, max_value=hi,
                              value=int(round(orig_val)), step=1,
                              key=f"wi_{feat}")
                )

    modified_df = pd.DataFrame([modified_values], columns=features)
    mod_pred = predict_model.predict(modified_df)[0]
    mod_prob = predict_model.predict_proba(modified_df)[0]

    changed_feats = [
        f for f in features if abs(modified_values[f] - float(original[f])) > 1e-9
    ]

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Original")
        st.metric("Prediction", class_names[orig_pred])
        st.metric("Probability of High Loneliness", f"{orig_prob[1]:.1%}")

        n_features = len(features)
        display_names_list = [get_display_name(f, display_names) for f in features]

        shap_explanation = shap.Explanation(
            values=shap_values_test[1][sample_idx],
            base_values=shap_expected_value[1]
                        if len(shap_expected_value) > 1
                        else float(shap_expected_value[0]),
            data=original.values,
            feature_names=display_names_list,
        )
        fig, _ = plt.subplots(figsize=(8, max(6, n_features * 0.25)))
        shap.waterfall_plot(shap_explanation, show=False, max_display=n_features)
        plt.title("Original Prediction Breakdown")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with right:
        st.subheader("Modified")
        pred_changed = mod_pred != orig_pred
        st.metric("Prediction", class_names[mod_pred],
                  delta="Prediction flipped!" if pred_changed else "No change",
                  delta_color="normal" if pred_changed else "off")
        st.metric("Probability of High Loneliness", f"{mod_prob[1]:.1%}",
                  delta=f"{mod_prob[1] - orig_prob[1]:+.1%}")

        if changed_feats:
            st.markdown("**What you changed:**")
            change_rows = []
            for f in changed_feats:
                change_rows.append({
                    "Feature": get_display_name(f, display_names),
                    "Original": format_feature_value(f, float(original[f]),
                                                     category_labels, likert_features),
                    "Modified": format_feature_value(f, modified_values[f],
                                                     category_labels, likert_features),
                })
            st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No features changed yet. Adjust the controls above.")

    if changed_feats:
        st.markdown("---")
        with st.expander("What does this mean?", expanded=True):
            shap_vals = shap_values_test[1][sample_idx]
            st.markdown(explain_whatif(
                original.to_dict(), modified_values, features, display_names,
                category_labels, likert_features, shap_vals,
                class_names, orig_pred, mod_pred, orig_prob, mod_prob
            ))


# ── Counterfactual Explorer Page ────────────────────────────────────────────

def render_counterfactuals(model, X_test, y_test, features, class_names,
                           feature_info, display_names, category_labels,
                           likert_features, precomputed_cfs, precomputed_preds):
    st.title("Counterfactual Explorer")

    st.info(
        "Counterfactual explanations answer: **\"What would need to change for this "
        "person's prediction to be different?\"** They show the smallest changes needed "
        "to flip the model's decision."
    )

    if precomputed_cfs is not None:
        available_indices = sorted(precomputed_cfs.keys())
        st.success(f"Loaded {len(available_indices)} pre-computed counterfactual scenarios.")
    else:
        available_indices = list(range(len(X_test)))
        st.warning("No pre-computed counterfactuals found. Run the notebook first.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Selected Person")

        person_labels = {
            i: build_person_label(i, X_test.iloc[i], class_names,
                                  display_names, category_labels, likert_features,
                                  precomputed_preds)
            for i in available_indices
        }
        sample_idx = st.selectbox(
            "Select a person:", available_indices,
            format_func=lambda x: person_labels[x],
            key="cf_sample"
        )

        instance = X_test.iloc[[sample_idx]]

        if precomputed_preds is not None and sample_idx < len(precomputed_preds["y_pred"]):
            pred = int(precomputed_preds["y_pred"][sample_idx])
            prob_high = float(precomputed_preds["y_prob"][sample_idx])
            prob = np.array([1.0 - prob_high, prob_high])
            st.markdown(f"**Current Prediction**: {class_names[pred]} "
                        f"({prob[pred]:.1%} confidence)")
        else:
            try:
                pred = model.predict(instance)[0]
                prob = model.predict_proba(instance)[0]
                st.markdown(f"**Current Prediction**: {class_names[pred]} "
                            f"({prob[pred]:.1%} confidence)")
            except Exception:
                pred = None
                prob = None
                st.warning("Could not compute prediction.")

        with st.expander("View this person's profile"):
            profile = []
            for feat in features:
                val = instance.iloc[0][feat]
                profile.append({
                    "Feature": get_display_name(feat, display_names),
                    "Value": format_feature_value(feat, val, category_labels, likert_features),
                })
            st.dataframe(pd.DataFrame(profile), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Counterfactual Scenarios")

        cf_records = None
        original_values = instance.iloc[0]

        if precomputed_cfs is not None and sample_idx in precomputed_cfs:
            cf_records = precomputed_cfs[sample_idx]
            st.markdown(f"Found **{len(cf_records)}** alternative scenario(s).")
        else:
            st.info("No counterfactuals available for this person. "
                    "Try a different person or run the notebook with more samples.")

        if cf_records is not None:
            st.markdown("**What would need to change?**")
            for i, cf in enumerate(cf_records):
                with st.expander(f"Scenario {i+1}", expanded=(i == 0)):
                    changes = []
                    for feat in features:
                        if feat in cf and feat in original_values:
                            orig_val = original_values[feat]
                            cf_val = cf[feat]
                            if abs(cf_val - orig_val) > 0.01:
                                changes.append({
                                    "Feature": get_display_name(feat, display_names),
                                    "Current": format_feature_value(
                                        feat, orig_val, category_labels, likert_features),
                                    "Needed": format_feature_value(
                                        feat, cf_val, category_labels, likert_features),
                                    "Direction": "Increase" if cf_val > orig_val else "Decrease"
                                })
                    if changes:
                        st.dataframe(pd.DataFrame(changes),
                                     use_container_width=True, hide_index=True)
                        st.markdown("---")
                        st.markdown(explain_counterfactual(
                            original_values.to_dict(), cf, features,
                            display_names, category_labels, likert_features,
                            class_names, pred if pred is not None else 0
                        ))
                    else:
                        st.info("No significant changes in this scenario.")

    # Summary across all counterfactuals
    if cf_records and len(cf_records) > 1:
        px, _ = _import_plotly()
        st.markdown("---")
        st.subheader("Summary: Most Frequently Changed Features")
        st.markdown(
            "Across all scenarios, these features were changed most often, "
            "suggesting they are the most important levers for this person:"
        )

        feat_counts = {}
        for cf in cf_records:
            for feat in features:
                if feat in cf and feat in original_values:
                    if abs(cf[feat] - original_values[feat]) > 0.01:
                        dname = get_display_name(feat, display_names)
                        feat_counts[dname] = feat_counts.get(dname, 0) + 1

        if feat_counts:
            summary_df = pd.DataFrame([
                {"Feature": k, "Times Changed": v}
                for k, v in sorted(feat_counts.items(), key=lambda x: -x[1])
            ])
            fig = px.bar(summary_df, x="Times Changed", y="Feature",
                         orientation="h", color_discrete_sequence=["steelblue"])
            fig.update_layout(yaxis=dict(autorange="reversed"),
                              height=max(300, len(feat_counts) * 30))
            st.plotly_chart(fig, use_container_width=True)
