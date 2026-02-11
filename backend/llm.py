# backend/llm.py

import requests
from backend.config import GROQ_API_KEY


def generate_with_llm(context: str, user_query: str) -> str:
    """
    Generates a supportive summary using Hosted LLM (Groq).
    Applies Self-RAG reflection before returning output.
    Falls back safely if API fails.
    """

    prompt = f"""
You are a compassionate mental health navigator with clinical awareness.

STRICT BOUNDARIES:
- No diagnostic labels or disorder names
- No medication recommendations
- Clinical tone without medical jargon
- Concise, structured responses (120–150 words)
- Use precise language that respects the user's experience

RESPONSE STRUCTURE (MANDATORY):

### Clinical Understanding
Briefly contextualize what they're experiencing using evidence-informed language (2 short points)

### Underlying Mechanisms
Explain potential psychophysiological or psychological factors at play (2 short points)

### Evidence-Based Self-Care Step
One specific, actionable intervention grounded in therapeutic principles (1 practical action)

### Professional Consultation Indicators
When these patterns warrant clinical evaluation (1 clear sentence)

CONTEXT:
{context}

USER INPUT:
{user_query}

TONE GUIDELINES:
- Use terms like "stress response," "cognitive patterns," "emotional regulation," "physiological symptoms"
- Validate without minimizing: "This suggests..." rather than "You might just be..."
- Balance warmth with clinical precision
- Convey competence without overpromising

RESPONSE:
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
               "model":"llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a supportive mental health assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.4
            },
            timeout=30
        )

        # 🔍 Debug status
        print("GROQ STATUS:", response.status_code)

        if response.status_code != 200:
            print("GROQ ERROR:", response.text)
            raise RuntimeError("Groq API failed")

        data = response.json()

        draft = data["choices"][0]["message"]["content"].strip()

        print("LLM RAW RESPONSE:", draft)

        if not draft:
            raise RuntimeError("Empty LLM response")

        return self_rag_reflection(draft)

    except Exception as e:
        print("LLM FALLBACK TRIGGERED:", str(e))
        return fallback_response(context, user_query)


# ================================
# SELF-RAG REFLECTION
# ================================
def self_rag_reflection(draft: str) -> str:

    banned_phrases = [
        "you have",
        "you are diagnosed",
        "this means you are",
        "disorder",
        "mental illness",
        "medication",
        "prescription",
        "treatment plan",
        "clinical condition"
    ]

    lowered = draft.lower()

    for phrase in banned_phrases:
        if phrase in lowered:
            return (
                "Based on what you shared, some patterns may be relevant.\n\n"
                "This tool does not provide diagnoses. Its purpose is to help "
                "you reflect on your experiences in a supportive way.\n\n"
                "If these concerns feel intense, persistent, or unsafe, "
                "seeking support from a qualified mental health professional "
                "or a trusted person is strongly recommended."
            )

    if "professional" not in lowered and "support" not in lowered:
        draft += (
            "\n\nIf these experiences start to feel overwhelming or unsafe, "
            "reaching out to a mental health professional or a trusted person "
            "can be very helpful."
        )

    return draft


# ================================
# FALLBACK
# ================================
def fallback_response(context: str, user_query: str) -> str:

    return (
        "Based on what you shared, there are some patterns that may be worth noticing.\n\n"
        f"{context[:500]}\n\n"
        "Many people experience similar feelings, especially during times of stress "
        "or change. This is not a diagnosis.\n\n"
        "If these experiences feel overwhelming or unsafe, seeking human support "
        "from a trusted person or professional is important."
    )
