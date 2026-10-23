"""MACHINE-LEARNING PATH: Random Forest + Logistic Regression trained on SYNTHETIC cases made from the rules."""
import difflib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from knowledge_base import RULES, SYMPTOMS, DISEASES

# small alias list: everyday words -> known symptom names
ALIASES = {"temperature": "fever", "tired": "fatigue", "tiredness": "fatigue", "ache": "body ache",
           "throwing up": "vomiting", "loose motion": "diarrhea", "breathless": "shortness of breath",
           "skin rash": "rash", "head ache": "headache", "joint ache": "joint pain"}

def to_vector(symptoms):
    """Symptom vector: 1 if the symptom is present, else 0 (one slot per known symptom)."""
    return [1 if s in symptoms else 0 for s in SYMPTOMS]

def make_training_data(seed=0):
    """Synthetic data: for each rule make full cases + partial cases (random subset of conditions)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _, conds, disease, _ in RULES:
        for _ in range(15):
            keep = list(conds) if rng.random() < 0.5 else [c for c in conds if rng.random() < 0.7] or [conds[0]]
            noise = list(rng.choice(SYMPTOMS, size=int(rng.integers(0, 2))))   # 0-1 random extra symptom
            X.append(to_vector(set(keep) | set(noise)))
            y.append(disease)
    return np.array(X), np.array(y)

def train_models():
    X, y = make_training_data()
    rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    return rf, lr

def predict(models, symptoms, top_n=5):
    """Blend RF + LR probabilities, then adjust using partial rule coverage. Returns ranked list."""
    rf, lr = models
    v = np.array([to_vector(symptoms)])
    blend = (rf.predict_proba(v)[0] + lr.predict_proba(v)[0]) / 2
    results = []
    for disease, p in zip(rf.classes_, blend):
        coverage = max(len([c for c in conds if c in symptoms]) / len(conds)
                       for _, conds, d, _ in RULES if d == disease)
        score = 0.7 * p + 0.3 * coverage * (1 if p > 0 else 0.5)   # blend model with rule coverage
        matched = sorted({c for _, conds, d, _ in RULES if d == disease for c in conds if c in symptoms})
        missing = sorted({c for _, conds, d, _ in RULES if d == disease for c in conds if c not in symptoms})
        if matched:
            results.append((disease, score, matched, missing[:4]))
    total = sum(r[1] for r in results) or 1
    results = [(d, s / total, m, x) for d, s, m, x in results]       # normalise to percentages
    return sorted(results, key=lambda r: -r[1])[:top_n]

def parse_text(text):
    """Free text -> known symptoms using aliases, substring match, then close string match."""
    text, found = text.lower(), set()
    for alias, symptom in ALIASES.items():
        if alias in text:
            found.add(symptom)
    for s in SYMPTOMS:
        if s in text:
            found.add(s)
    for word in text.replace(",", " ").split():
        found.update(difflib.get_close_matches(word, SYMPTOMS, n=1, cutoff=0.85))
    return sorted(found)
