# backend/main.py

from fastapi import FastAPI
from backend.schemas import UserInput, NavigatorResponse
from backend.safety import check_risk
from backend.retriever import KBRetriever
from backend.llm import generate_with_llm


app = FastAPI(title="Mind Rewire Navigator")

retriever = KBRetriever()


# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def health_check():
    return {"status": "ok"}


# ===============================
# NAVIGATOR ENDPOINT
# ===============================
@app.post("/navigator")
def navigator(input: UserInput):

    # Step 1 — Safety check
    risk = check_risk(input.text)

    if risk == "HIGH":
        return NavigatorResponse(
            status="CRISIS",
            summary="Please seek immediate help. You are not alone.",
            focus_areas=[],
            plan_today=[],
            plan_week=[]
        )

    # Step 2 — Retrieve RAG context
    context_chunks = retriever.search(input.text, top_k=1)
    context = "\n\n".join(context_chunks)

    # Step 3 — Generate summary via Hosted LLM
    summary = generate_with_llm(context, input.text)

    # ===============================
    # Step 4 — CONTROLLED FOCUS EXTRACTION
    # ===============================
    text_lower = input.text.lower()
    focus_areas = []

    if any(word in text_lower for word in ["sleep", "insomnia", "restless"]):
        focus_areas.append("Sleep patterns")

    if any(word in text_lower for word in ["anxiety", "anxious", "panic", "worry"]):
        focus_areas.append("Anxiety & worry")

    if any(word in text_lower for word in ["stress", "overwhelm", "pressure"]):
        focus_areas.append("Stress levels")

    if any(word in text_lower for word in ["sad", "low", "empty", "down"]):
        focus_areas.append("Low mood")

    if not focus_areas:
        focus_areas.append("Emotional wellbeing")

    # ===============================
    # Step 5 — CONTROLLED PLAN TEMPLATES
    # ===============================

    plan_today = [
        "Pause and take 5 slow, deep breaths.",
        "Write down one thought that feels heavy right now."
    ]

    # 7-day structured plan
    plan_week = [
        "Create a calming wind-down routine before sleep.",
        "Schedule one small activity you usually enjoy.",
        "Notice repeating thoughts without judging them.",
        "Take short breaks when feeling overwhelmed.",
        "Practice grounding using your senses.",
        "Spend time outdoors or in natural light.",
        "Reflect on what felt slightly better this week."
    ]

    # ===============================
    # FINAL RESPONSE
    # ===============================
    return NavigatorResponse(
        status="SAFE",
        summary=summary,
        focus_areas=focus_areas,
        plan_today=plan_today,
        plan_week=plan_week
    )
