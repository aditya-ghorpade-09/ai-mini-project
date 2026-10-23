"""KNOWLEDGE BASE: rules stored as plain data (report: 'rules defined as data').
NOTE: Only R01 (influenza, CF 0.90) and the dengue/malaria/typhoid/asthma examples come from the
report. The other rules are SAMPLE rules written for this demo - not clinically validated."""

# (rule id, [conditions], conclusion, certainty factor)
RULES = [
    ("R01", ["fever", "cough", "sore throat", "body ache"], "Influenza", 0.90),
    ("R02", ["fever", "chills", "fatigue"], "Influenza", 0.75),
    ("R03", ["runny nose", "sneezing", "sore throat"], "Common Cold", 0.85),
    ("R04", ["runny nose", "cough", "mild fever"], "Common Cold", 0.70),
    ("R05", ["fever", "dry cough", "loss of smell"], "COVID-19", 0.90),
    ("R06", ["fever", "fatigue", "shortness of breath"], "COVID-19", 0.75),
    ("R07", ["high fever", "severe headache", "joint pain", "rash"], "Dengue", 0.90),
    ("R08", ["high fever", "joint pain", "eye pain"], "Dengue", 0.75),
    ("R09", ["chills", "cyclical fever", "sweating", "headache"], "Malaria", 0.90),
    ("R10", ["chills", "cyclical fever", "muscle pain"], "Malaria", 0.70),
    ("R11", ["prolonged fever", "abdominal pain", "loss of appetite", "weakness"], "Typhoid", 0.85),
    ("R12", ["wheezing", "chest tightness", "shortness of breath"], "Asthma", 0.90),
    ("R13", ["wheezing", "cough"], "Asthma", 0.65),
    ("R14", ["fever", "productive cough", "chest pain", "shortness of breath"], "Pneumonia", 0.85),
    ("R15", ["severe headache", "nausea", "light sensitivity"], "Migraine", 0.85),
    ("R16", ["diarrhea", "vomiting", "abdominal pain"], "Gastroenteritis", 0.85),
    ("R17", ["prolonged cough", "night sweats", "weight loss"], "Tuberculosis", 0.85),
    ("R18", ["sneezing", "itchy eyes", "runny nose"], "Allergic Rhinitis", 0.85),
    ("R19", ["rash", "fever", "itching"], "Chickenpox", 0.80),
    ("R20", ["prolonged fever", "abdominal pain", "weakness"], "Typhoid", 0.70),
]

SYMPTOMS = sorted({c for _, conds, _, _ in RULES for c in conds})
DISEASES = sorted({d for _, _, d, _ in RULES})

# Symptom groups used only to organise the checklist in the interface
CATEGORIES = {
    "Fever & General Vitals": ["fever", "high fever", "mild fever", "cyclical fever", "prolonged fever", "chills",
                               "sweating", "fatigue", "weakness", "body ache", "muscle pain", "loss of appetite"],
    "Respiratory & Chest": ["cough", "dry cough", "productive cough", "prolonged cough", "sore throat", "runny nose",
                            "sneezing", "shortness of breath", "wheezing", "chest tightness", "chest pain"],
    "Head, Eyes & Skin": ["headache", "severe headache", "light sensitivity", "eye pain", "itchy eyes", "rash",
                          "itching", "loss of smell"],
    "Digestive & Other": ["abdominal pain", "nausea", "vomiting", "diarrhea", "joint pain", "night sweats", "weight loss"],
}

def all_diseases():
    """Recomputed each time so a rule added from the sidebar is included."""
    return sorted({d for _, _, d, _ in RULES})
