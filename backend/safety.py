# backend/safety.py

CRISIS_KEYWORDS = [
    "kill myself",
    "suicide",
    "end my life",
    "hurt myself",
    "can't go on"
]

def check_risk(text: str) -> str:
    text = text.lower()
    for word in CRISIS_KEYWORDS:
        if word in text:
            return "HIGH"
    return "LOW"
