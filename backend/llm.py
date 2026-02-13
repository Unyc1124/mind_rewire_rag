# backend/llm.py

import requests
from backend.config import GROQ_API_KEY


def generate_with_llm(context: str, user_query: str) -> str:
    """
    Generates a supportive summary using Hosted LLM (Groq).
    Applies Self-RAG reflection before returning output.
    Falls back safely if API fails.
    
    Memory-optimized for Render deployment.
    """
    
    # Truncate context to prevent memory bloat (max ~500 chars)
    context = context[:500] if context else ""
    user_query = user_query[:300] if user_query else ""

    prompt = f"""You are a compassionate mental health navigator with clinical awareness.

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

RESPONSE:"""

    try:
        # Use session for connection pooling (reduces memory overhead)
        with requests.Session() as session:
            response = session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
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
                    "temperature": 0.4,
                    "max_tokens": 300,  # Limit response length to reduce memory
                    "stream": False
                },
                timeout=20  # Reduced timeout
            )

        # Check status before parsing JSON
        if response.status_code != 200:
            print(f"GROQ ERROR ({response.status_code}): {response.text[:200]}")
            return fallback_response(context, user_query)

        # Parse JSON and immediately extract needed data
        data = response.json()
        draft = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Clear response object to free memory
        del response
        del data

        if not draft:
            return fallback_response(context, user_query)

        # Apply reflection and return
        return self_rag_reflection(draft)

    except requests.exceptions.Timeout:
        print("LLM TIMEOUT - using fallback")
        return fallback_response(context, user_query)
    
    except requests.exceptions.RequestException as e:
        print(f"LLM REQUEST ERROR: {str(e)[:100]}")
        return fallback_response(context, user_query)
    
    except (KeyError, IndexError, ValueError) as e:
        print(f"LLM PARSE ERROR: {str(e)[:100]}")
        return fallback_response(context, user_query)
    
    except Exception as e:
        print(f"LLM UNEXPECTED ERROR: {str(e)[:100]}")
        return fallback_response(context, user_query)


# ================================
# SELF-RAG REFLECTION
# ================================
def self_rag_reflection(draft: str) -> str:
    """
    Validates LLM output against safety guidelines.
    Memory-optimized with early returns.
    """
    
    # Truncate if too long
    if len(draft) > 1000:
        draft = draft[:1000] + "..."
    
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

    # Early return if banned phrase detected
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

    # Add professional support reminder if missing
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
    """
    Provides safe fallback when LLM fails.
    Memory-efficient with length limits.
    """
    
    # Truncate context to prevent memory bloat
    context_snippet = context[:300] if context else "your experiences"
    
    return (
        "Based on what you shared, there are some patterns that may be worth noticing.\n\n"
        "Many people experience similar feelings, especially during times of stress "
        "or change. This is not a diagnosis.\n\n"
        "If these experiences feel overwhelming or unsafe, seeking human support "
        "from a trusted person or professional is important.\n\n"
        "Consider: What specific situations trigger these feelings? "
        "Are there times when you feel more at ease?"
    )