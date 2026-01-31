# backend/main.py

from fastapi import FastAPI
from backend.schemas import UserInput, NavigatorResponse
from backend.safety import check_risk
from backend.kb_loader import load_kb_sections
from backend.embeddings import build_kb_embeddings
from backend.retriever import KBRetriever
from backend.llm import generate_with_llm


app = FastAPI(title="Mind Rewire Navigator")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/debug/embeddings")
def debug_embeddings():
    chunks, embeddings = build_kb_embeddings()
    return {
        "total_chunks": len(chunks),
        "embedding_shape": len(embeddings[0]),
        "sample_text": chunks[0][:200]
    }



@app.get("/debug/kb")
def debug_kb():
    chunks = load_kb_sections()
    return {
        "total_chunks": len(chunks),
        "sample_chunk": chunks[0][:300]
    }


retriever = KBRetriever()

@app.get("/debug/search")
def debug_search(q: str):
    results = retriever.search(q)
    return {
        "query": q,
        "matches": results
    }

# @app.post("/navigator")
# def navigator(input: UserInput):
#     # Step 1: safety check
#     risk = check_risk(input.text)

#     if risk == "HIGH":
#         return NavigatorResponse(
#             status="CRISIS",
#             message="Please seek immediate help. You are not alone."
#         )

#     # Step 2: placeholder (we add RAG later)
#     return NavigatorResponse(
#         status="SAFE",
#         message="Thanks for sharing. We are building your support plan."
#     )


@app.post("/navigator")
def navigator(input: UserInput):
    # Step 1: Safety check
    risk = check_risk(input.text)

    if risk == "HIGH":
        return NavigatorResponse(
            status="CRISIS",
            message="Please seek immediate help. You are not alone."
        )

    # Step 2: Retrieve ONLY top 2 relevant KB chunks
    context_chunks = retriever.search(input.text, top_k=1)
    context = "\n\n".join(context_chunks)

    # Step 3: Generate response using LLM + Self-RAG
    answer = generate_with_llm(context, input.text)

    return NavigatorResponse(
        status="SAFE",
        message=answer
    )
