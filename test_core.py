import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from knowledge_base import RULES, DISEASES
from inference import combine_cf, forward_chaining, backward_chaining
import ml_predictor

def test_kb_size():
    assert len(RULES) == 20 and len(DISEASES) == 13

def test_cf_example_from_report():
    assert abs(combine_cf(0.90, 0.75) - 0.975) < 1e-9

def test_forward_r01():
    concl, _ = forward_chaining(["fever", "cough", "sore throat", "body ache"])
    assert concl["Influenza"] == 0.90

def test_backward_proved_and_rejected():
    ok = {"high fever": True, "severe headache": True, "joint pain": True, "rash": True}
    assert backward_chaining("Dengue", ok)[0] == 0.90
    assert backward_chaining("Dengue", {})[0] is None

def test_ml_and_text():
    assert "fever" in ml_predictor.parse_text("I have fever and feel tired")
    top = ml_predictor.predict(ml_predictor.train_models(), {"chills", "cyclical fever", "muscle pain"})[0][0]
    assert top == "Malaria"
