"""FastAPI web server for docagent: a JSON API + a static chat UI.

Run:
    python -m docagent.web              # then open http://127.0.0.1:8000
    # or: uvicorn docagent.web:app --reload

Endpoints:
    POST /api/ask      {question} -> {intent, answer, citations, trace}
    GET  /api/sources  -> {sources: [...]}
    GET  /             -> the chat UI (static/index.html)
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from docagent.agent import docagent  # noqa: E402  (load_dotenv must run first)
from docagent.retriever import get_retriever  # noqa: E402

app = FastAPI(
    title="docagent",
    description="Agentic RAG over local documents, with citations.",
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    intent: str
    answer: str
    citations: list[str]
    trace: list[dict]


def _extract(result: dict):
    intent = result.get("classification_decision", "") or ""
    answer, citations = "", []
    for msg in reversed(result.get("messages", [])):
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc["name"] == "Answer":
                answer = tc["args"].get("answer", "")
                citations = tc["args"].get("citations", []) or []
                break
        if answer:
            break
    if not answer and result.get("messages"):
        answer = str(result["messages"][-1].content)
    return intent, answer, citations


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = docagent.invoke(
        {"question_input": {"question": req.question}},
        config={"recursion_limit": 12},
    )
    intent, answer, citations = _extract(result)
    return AskResponse(
        intent=intent,
        answer=answer,
        citations=citations,
        trace=result.get("trace", []) or [],
    )


@app.get("/api/sources")
def sources() -> dict:
    return {"sources": get_retriever().list_sources()}


# Static chat UI (mounted last so /api/* routes take precedence).
_STATIC = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="ui")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
