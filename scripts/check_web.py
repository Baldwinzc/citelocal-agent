#!/usr/bin/env python
"""Smoke-test the web API in-process (needs an LLM API key for /api/ask).

    python scripts/check_web.py
"""

from fastapi.testclient import TestClient

from docagent.web import app

c = TestClient(app)


def main():
    r = c.get("/")
    print(f"GET /            -> {r.status_code}  has-ui={'docagent' in r.text}")

    r = c.get("/api/sources")
    print(f"GET /api/sources -> {r.status_code}  {len(r.json().get('sources', []))} docs")

    r = c.post("/api/ask", json={"question": "How do I declare an integer path parameter?"})
    d = r.json()
    print(f"POST /api/ask    -> {r.status_code}  intent={d.get('intent')}")
    print("  answer  :", (d.get("answer") or "")[:180])
    print("  cites   :", d.get("citations"))
    print("  trace   :", [t.get("query") for t in d.get("trace", []) if t.get("step") == "search_docs"])


if __name__ == "__main__":
    main()
