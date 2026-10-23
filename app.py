"""USER INTERFACE (Streamlit). Run with:  streamlit run app.py"""
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import knowledge_base as kb
from knowledge_base import RULES, SYMPTOMS, CATEGORIES
from inference import forward_chaining, backward_chaining, partial_matches
import ml_predictor

TEAM = "Sarvesh Pol, Harshal Andhale, Kartik Fatangade, Aditya Ghorpade, Krushna Indure"  # change names here
st.set_page_config(page_title="MediLogix", page_icon="🩺", layout="wide")

# ---------- styling (pills, cards, gradient bars) ----------
st.markdown("""<style>
.block-container {max-width: 1000px; padding-top: 3rem;}
.hero {text-align:center; margin-bottom: 1.2rem;}
.hero h1 {color:#3b5bdb; font-size:2.1rem; margin:0;}
.hero p {color:#475569; font-size:0.8rem; margin:2px 0 0 0;}
.pill {display:inline-block; padding:2px 12px; border-radius:14px; font-size:0.78rem; margin:3px 4px 3px 0;}
.green {background:#dcfce7; color:#166534; border:1px solid #86efac;}
.red {background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5;}
.selbox {background:#f1f5f9; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px; min-height:48px; color:#64748b; font-style:italic;}
.confirm {background:#e6faf0; border:1px solid #bbf7d0; border-left:5px solid #10b981; border-radius:10px; padding:16px 20px;
          display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;}
.reject {background:#fef2f2; border:1px solid #fecaca; border-left:5px solid #ef4444;}
.confirm small {color:#047857; font-weight:700; letter-spacing:.5px; font-size:.68rem;}
.confirm h2 {margin:2px 0; color:#047857;} .confirm p {margin:0; color:#047857; font-size:.78rem;}
.cfbox {background:white; border-radius:8px; padding:8px 18px; text-align:center; box-shadow:0 1px 4px #0002; color:#047857;}
.cfbox b {font-size:1.6rem; display:block;}
.bar {height:9px; border-radius:6px; background:#e2e8f0; margin:8px 0;}
.bar div {height:9px; border-radius:6px; background:linear-gradient(90deg,#2a7de1,#10b981);}
.mlcard {background:white; border:1px solid #e2e8f0; border-left:5px solid #3b82f6; border-radius:10px; padding:14px 18px; margin-bottom:14px; box-shadow:0 1px 4px #0001;}
.mlcard .top {display:flex; justify-content:space-between; font-weight:700;} .mlcard .pct {color:#2563eb;}
.mlcard .row {font-size:.78rem; color:#334155; margin-top:4px;}
</style>""", unsafe_allow_html=True)

@st.cache_resource            # train the ML models only once
def get_models():
    return ml_predictor.train_models()

def name(s): return s.title()
def pills(items, cls): return "".join(f"<span class='pill {cls}'>{name(i)}</span>" for i in items) or "<i>None</i>"

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("<div style='font-size:3.5rem'>🧑‍⚕️</div>", unsafe_allow_html=True)
    st.subheader("⚙️ MediLogix")
    mode = st.radio("Select inference mode", ["🔵 Forward Chaining", "🟢 Backward Chaining",
                                              "🟣 ML & Random Forest Symptom Predictor"])
    st.divider()
    with st.expander("📖 About the modes"):
        st.write("**Forward:** symptoms → rules → diagnoses.\n\n**Backward:** pick a disease and test whether its rules are satisfied.\n\n"
                 "**ML:** Random Forest + Logistic Regression ranking (trained on synthetic data).\n\n"
                 "CF = CF1 + CF2 × (1 − CF1). Educational use only; not medical advice.")
    st.divider()
    with st.expander("➕ Add a new rule"):
        new_d = st.text_input("Disease name")
        new_c = st.multiselect("IF (all conditions)", SYMPTOMS, format_func=name)
        new_cf = st.slider("Certainty factor", 0.1, 1.0, 0.8, 0.05)
        if st.button("Add rule"):
            if not new_d.strip() or len(new_c) < 2:
                st.error("Enter a disease and at least 2 conditions.")
            else:
                RULES.append((f"R{len(RULES)+1:02d}", new_c, new_d.strip().title(), new_cf))  # in-memory only
                st.cache_resource.clear()            # retrain ML with the new rule
                st.success("Rule added (until the app restarts).")

st.markdown("<div class='hero'><h1>🩺 MediLogix</h1><p>Rule-Based Medical Diagnosis Expert System · "
            "Forward &amp; Backward Chaining · MYCIN Certainty Factors</p></div>", unsafe_allow_html=True)

# ---------------- FORWARD ----------------
if mode.startswith("🔵"):
    st.header("🔵 Forward Chaining — Symptom Checker")
    st.write("Select your symptoms from the checklist below. The engine applies expert rules with MYCIN certainty factors, "
             "backed by a **Machine Learning fallback** for partial symptom combinations.")
    selected = [s for s in SYMPTOMS if st.session_state.get(f"fw_{s}")]
    c1, c2 = st.columns([4, 1])
    c1.markdown("<div class='selbox'>" + (pills(selected, "green") if selected else
                "No symptoms selected yet. Check symptoms from the categories below:") + "</div>", unsafe_allow_html=True)
    def clear_all():
        for s in SYMPTOMS: st.session_state[f"fw_{s}"] = False
    c2.button("🗑 Clear All", on_click=clear_all, use_container_width=True)
    for cat, items in CATEGORIES.items():
        with st.expander(f"🌡 {cat} ({len(items)} symptoms)", expanded=(cat == "Fever & General Vitals")):
            cols = st.columns(3)
            for i, s in enumerate(items):
                cols[i % 3].checkbox(name(s), key=f"fw_{s}")
    if st.button("🔍 Run diagnosis", type="primary"):
        if not selected:
            st.error("Please select at least one symptom.")
        else:
            conclusions, trace = forward_chaining(selected)
            if conclusions:
                for d, cf in sorted(conclusions.items(), key=lambda x: -x[1]):
                    st.markdown(f"<div class='confirm'><div><small>DIAGNOSIS DERIVED</small><h2>{d}</h2>"
                                f"<p>All conditions of at least one rule are satisfied.</p></div>"
                                f"<div class='cfbox'><small>CERTAINTY FACTOR</small><b>{cf:.0%}</b>MYCIN-Combined</div></div>",
                                unsafe_allow_html=True)
            else:
                st.warning("No complete rule fired — showing partial rule matches and the ML ranking instead.")
                for rid, d, have, miss in partial_matches(selected):
                    st.markdown(f"**{d}** ({rid}) — matched {pills(have,'green')} missing {pills(miss,'red')}", unsafe_allow_html=True)
                for d, score, m, x in ml_predictor.predict(get_models(), set(selected), 3):
                    st.write(f"ML: **{d}** — {score:.0%} match")
            with st.expander("🔎 Why? — Full Reasoning Trace", expanded=True):
                for line in trace or ["No rules fired."]:
                    st.write("• " + line)
    with st.expander("📚 Knowledge base & inference network"):
        st.dataframe(pd.DataFrame([{"Rule": r, "IF (all of)": ", ".join(c), "THEN": d, "CF": cf}
                                   for r, c, d, cf in RULES]), use_container_width=True, hide_index=True)
        if st.button("Draw inference network"):
            g = nx.DiGraph()
            for rid, conds, d, _ in RULES:
                for c in conds: g.add_edge(c, rid)
                g.add_edge(rid, d)
            fig, ax = plt.subplots(figsize=(11, 8))
            ds = kb.all_diseases()
            colors = ["#7fb8e6" if n in SYMPTOMS else "#f4c27a" if n in ds else "#cccccc" for n in g]
            nx.draw(g, nx.spring_layout(g, seed=3, k=0.6), ax=ax, node_color=colors, with_labels=True, font_size=6, node_size=500)
            st.pyplot(fig)
            st.caption("Blue = symptoms, grey = rules, orange = conditions (diseases).")

# ---------------- BACKWARD ----------------
elif mode.startswith("🟢"):
    st.header("🟢 Backward Chaining — Hypothesis Tester")
    st.write("Choose a disease hypothesis. The engine finds the rules that conclude it and asks about the symptoms they need.")
    goal = st.selectbox("Disease hypothesis", kb.all_diseases())
    needed = sorted({c for _, conds, d, _ in RULES if d == goal for c in conds})
    st.write("**Do you have these symptoms?** (unticked = No)")
    cols = st.columns(3)
    answers = {s: cols[i % 3].checkbox(name(s), key=f"bw_{goal}_{s}") for i, s in enumerate(needed)}
    if st.button("🎯 Test hypothesis", type="primary"):
        cf, trace = backward_chaining(goal, answers)
        if cf:
            st.markdown(f"<div class='confirm'><div><small>DIAGNOSTIC HYPOTHESIS CONFIRMED</small><h2>🎯 {goal}</h2>"
                        f"<p>Patient answers satisfy all diagnostic conditions required for this diagnosis.</p></div>"
                        f"<div class='cfbox'><small>CERTAINTY FACTOR</small><b>{cf:.0%}</b>MYCIN-Combined</div></div>"
                        f"<div class='bar'><div style='width:{cf:.0%}'></div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='confirm reject'><div><small style='color:#b91c1c'>HYPOTHESIS REJECTED</small>"
                        f"<h2 style='color:#b91c1c'>{goal}</h2><p style='color:#b91c1c'>No rule for this disease is fully satisfied.</p></div></div>",
                        unsafe_allow_html=True)
        with st.expander("🗒 All Collected Answers Summary", expanded=True):
            a, b = st.columns(2)
            a.markdown("**Confirmed Symptoms (Yes):**<br>" + pills([s for s in needed if answers[s]], "green"), unsafe_allow_html=True)
            b.markdown("**Denied Symptoms (No):**<br>" + pills([s for s in needed if not answers[s]], "red"), unsafe_allow_html=True)
        with st.expander("🔎 Why? — Full Reasoning Trace", expanded=True):
            st.markdown("##### 🧭 Step-by-Step Rule Inference Chain")
            for line in trace:
                st.write("• " + line)

# ---------------- ML ----------------
else:
    st.header("🟣 ML & Random Forest Symptom Predictor")
    st.write("Select symptoms and/or type them freely. Random Forest and Logistic Regression probabilities are blended "
             "(trained on **synthetic** data generated from the rules).")
    chosen = st.multiselect("Select symptoms", SYMPTOMS, format_func=name)
    text = st.text_input("…or type symptoms freely (e.g. 'fever, tired and bad headache')")
    if st.button("🔮 Predict", type="primary"):
        symptoms = set(chosen) | set(ml_predictor.parse_text(text))
        if not symptoms:
            st.error("Enter or select at least one recognised symptom.")
        else:
            st.markdown("**Recognised symptoms:** " + pills(sorted(symptoms), "green"), unsafe_allow_html=True)
            st.subheader("⊕ Probabilistic & ML Diagnoses")
            for d, score, matched, missing in ml_predictor.predict(get_models(), symptoms):
                st.markdown(f"<div class='mlcard'><div class='top'><span>🟡 {d}</span><span class='pct'>{score:.0%} match</span></div>"
                            f"<div class='bar'><div style='width:{min(score,1):.0%}'></div></div>"
                            f"<div class='row'><b>Matched Symptoms:</b> {pills(matched,'green')}</div>"
                            f"<div class='row'><b>Key Missing Symptoms to Confirm:</b> {pills(missing,'red')}</div></div>",
                            unsafe_allow_html=True)

st.caption(f"Educational use only — rules and training data are not clinically validated. Team: {TEAM}")
