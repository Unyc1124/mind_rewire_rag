# backend/llm.py

import subprocess


def generate_with_llm(context: str, user_query: str) -> str:
    """
    Generates a response using a local Ollama LLM.
    Applies a self-reflection (Self-RAG) safety pass before returning output.
    Falls back safely if the LLM fails.
    """

    prompt = f"""
You are a mental health navigator.

STRICT RULES (DO NOT BREAK):
- Do NOT diagnose
- Do NOT list all symptom clusters
- Focus ONLY on the most relevant patterns
- Write for a real user, not documentation
- Keep the response short and structured

YOU MUST FOLLOW THIS STRUCTURE:

1. What this might mean (2–3 sentences)
   - Mention only the most relevant patterns

2. Why this can happen (2–3 sentences)
   - Simple explanation, no clinical terms

3. One small thing to try today (1 step, practical)

4. When to seek help (1 sentence, gentle)

CONTEXT (reference only what is relevant):
{context}

USER INPUT:
{user_query}

FINAL RESPONSE:
"""


    try:
        result = subprocess.run(
            ["ollama", "run", "phi3:mini"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=30
        )

        draft = result.stdout.strip()

        # Apply Self-RAG reflection
        return self_rag_reflection(draft)

    except Exception:
        # Safe fallback if Ollama fails
        return fallback_response(context)


def self_rag_reflection(draft: str) -> str:
    """
    Self-RAG reflection layer.
    Ensures the response is non-diagnostic, safe, and spec-compliant.
    """

    banned_phrases = [
        "you have",
        "you are diagnosed",
        "this means you are",
        "disorder",
        "medication",
        "prescription",
        "treatment plan"
    ]

    lowered = draft.lower()

    # If unsafe/diagnostic language detected → rewrite safely
    for phrase in banned_phrases:
        if phrase in lowered:
            return (
                "Based on what you shared, some patterns may be relevant.\n\n"
                "This tool does not provide diagnoses, but helps people understand "
                "their experiences in a clearer way.\n\n"
                "If these concerns feel intense, persistent, or unsafe, "
                "seeking support from a qualified mental health professional is strongly recommended."
            )

    # Ensure help-seeking guidance is present
    if "professional" not in lowered and "support" not in lowered:
        draft += (
            "\n\nIf these experiences start to feel overwhelming or unsafe, "
            "reaching out to a mental health professional or trusted person can be very helpful."
        )

    return draft


def fallback_response(context: str) -> str:
    """
    Deterministic, safe fallback when LLM is unavailable.
    Still RAG-based and compliant with the spec.
    """

    return (
        "Based on what you shared, here are some patterns that may be relevant:\n\n"
        f"{context[:600]}\n\n"
        "Many people experience similar patterns, especially during periods of stress.\n"
        "This is not a diagnosis. If these concerns feel overwhelming or unsafe, "
        "seeking human support is important."
    )
