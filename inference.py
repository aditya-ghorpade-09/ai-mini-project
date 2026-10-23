"""INFERENCE ENGINE + EXPLANATION TRACER: forward chaining, backward chaining, CF combination."""
from knowledge_base import RULES
from working_memory import WorkingMemory

def combine_cf(cf1, cf2):
    """MYCIN-style combination: CF = CF1 + CF2 * (1 - CF1). Example: 0.90, 0.75 -> 0.975"""
    return cf1 + cf2 * (1 - cf1)

def forward_chaining(symptoms):
    """Data-driven: start from symptoms, fire every rule whose conditions are ALL true.
    Returns (conclusions {disease: combined CF}, trace [text lines])."""
    memory = WorkingMemory()
    for s in symptoms:                       # Step 1: load symptoms as facts
        memory.add(s)
    fired, cf_by_disease, trace = set(), {}, []
    new_rule_fired = True
    while new_rule_fired:                    # Step 4: repeat until a full pass fires nothing
        new_rule_fired = False
        for rule_id, conds, disease, cf in RULES:          # Step 3: scan every unfired rule
            if rule_id in fired or not all(memory.is_true(c) for c in conds):
                continue
            fired.add(rule_id)               # each rule fires only once
            new_rule_fired = True
            old = cf_by_disease.get(disease)
            cf_by_disease[disease] = cf if old is None else combine_cf(old, cf)
            msg = f"{rule_id} fired: IF {', '.join(conds)} THEN {disease} (rule CF {cf:.2f})"
            if old is not None:
                msg += f" -> combined CF {old:.2f} + {cf:.2f} = {cf_by_disease[disease]:.3f}"
            trace.append(msg)
    # a conclusion is asserted only when its combined CF is at least 0.5
    conclusions = {d: v for d, v in cf_by_disease.items() if v >= 0.5}
    return conclusions, trace

def partial_matches(symptoms, min_fraction=0.5):
    """Used when no complete rule fires: rules that are at least half satisfied."""
    out = []
    for rule_id, conds, disease, cf in RULES:
        have = [c for c in conds if c in symptoms]
        if have and len(have) < len(conds) and len(have) / len(conds) >= min_fraction:
            out.append((rule_id, disease, have, [c for c in conds if c not in symptoms]))
    return out

def backward_chaining(goal, answers):
    """Goal-driven: try to PROVE one disease. `answers` = {symptom: True/False} the user gave.
    A rule proves the goal if all its conditions are true; otherwise the goal is rejected."""
    trace, proved_cf = [], None
    candidates = [r for r in RULES if r[2] == goal]
    trace.append(f"Goal: prove '{goal}'. Rules concluding it: {', '.join(r[0] for r in candidates)}")
    for rule_id, conds, _, cf in candidates:
        checks = [(c, answers.get(c, False)) for c in conds]
        shown = ", ".join(f"{c}={'YES' if v else 'NO'}" for c, v in checks)
        if all(v for _, v in checks):
            trace.append(f"{rule_id}: {shown} -> all conditions satisfied (CF {cf:.2f})")
            proved_cf = cf if proved_cf is None else combine_cf(proved_cf, cf)
        else:
            trace.append(f"{rule_id}: {shown} -> failed")
    trace.append("Result: goal PROVED" if proved_cf else "Result: goal REJECTED (no rule satisfied)")
    return proved_cf, trace
