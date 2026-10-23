"""WORKING MEMORY: the set of facts known about the current user (symptom -> True/False)."""

class WorkingMemory:
    def __init__(self):
        self.facts = {}                      # e.g. {"fever": True, "rash": False}

    def add(self, symptom, value=True):
        self.facts[symptom] = value

    def is_true(self, symptom):
        return self.facts.get(symptom) is True

    def is_known(self, symptom):
        return symptom in self.facts
