"""
web_router.py

Adds LLM-decided routing to DocMind: before generating a final answer,
Groq is asked a cheap classification question — "can this be answered
from the retrieved PDF context, or does it need live web info?" — and
only calls the DuckDuckGo MCP server when needed.

Drop this alongside web_search_client.py. Wire `answer_question()` into
wherever your Streamlit app currently calls the LLM after ChromaDB
retrieval.

Requires:
    pip install groq
    (mcp + duckduckgo-mcp-server already installed from the previous step)
"""

import asyncio
import json
import os
from dotenv import load_dotenv
load_dotenv()  
from groq import Groq

from web_search_client import WebSearchClient

# Fast/cheap model for the routing decision itself — no need for your main
# answer-generation model here, this call should be near-instant.
# (Groq deprecated the old llama-3.x chat models — gpt-oss-20b is the
# current lightweight replacement.)
ROUTER_MODEL_NAME = "openai/gpt-oss-20b"

# Main model used to generate the actual answer shown to the user.
ANSWER_MODEL_NAME = "openai/gpt-oss-120b"

ROUTER_PROMPT = """You are a routing classifier for a document Q&A system.

The user asked a question. Below is the context retrieved from their PDF
documents. Decide whether this context is sufficient to answer the
question, or whether the question needs current/live information that
would not be in a static PDF (e.g. today's date, recent news, current
prices, "latest" anything, real-time data).

Question: {question}

Retrieved PDF context:
---
{context}
---

Respond with ONLY a JSON object, no other text:
{{"needs_web_search": true or false, "reason": "one short sentence"}}
"""

# A single shared client, built from GROQ_API_KEY in the environment
# (loaded from .env via python-dotenv in your app, same as FinSight Agent).
_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def _needs_web_search(question: str, pdf_context: str) -> tuple[bool, str]:
    """Synchronous call to Groq to decide if web search is needed."""
    prompt = ROUTER_PROMPT.format(question=question, context=pdf_context or "(no relevant context found)")

    completion = _client.chat.completions.create(
        model=ROUTER_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = completion.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps the JSON in them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw.replace("json", "", 1).strip()

    try:
        parsed = json.loads(raw)
        return bool(parsed.get("needs_web_search", False)), parsed.get("reason", "")
    except (json.JSONDecodeError, IndexError):
        # Fail safe: if the router response is malformed, don't block the
        # user's answer on it — just skip web search this time.
        return False, "router response could not be parsed"


def _run_web_search(question: str) -> str:
    """Bridges the async MCP client into DocMind's sync Streamlit flow."""

    async def _search():
        async with WebSearchClient() as client:
            return await client.search(question, max_results=5)

    return asyncio.run(_search())


def answer_question(question: str, pdf_context: str, chat_history: str = "") -> dict:
    """
    Main entry point for DocMind's app.py.

    1. Asks Groq whether the PDF context is enough to answer.
    2. If not, runs a live web search via the MCP server and merges results.
    3. Generates the final answer, tagged with which source(s) were used.

    Args:
        chat_history: optional recent conversation turns, formatted as plain
            text (e.g. "User: ...\\nAssistant: ...\\n"), so follow-up
            questions like "what about page 2?" still have context.

    Returns:
        {
            "answer": str,
            "used_web_search": bool,
            "router_reason": str,
        }
    """
    needs_web, reason = _needs_web_search(question, pdf_context)

    combined_context = pdf_context
    if needs_web:
        web_results = _run_web_search(question)
        combined_context = (
            f"--- From your documents ---\n{pdf_context or '(nothing relevant found)'}\n\n"
            f"--- From a live web search ---\n{web_results}"
        )

    history_block = f"Recent conversation:\n{chat_history}\n\n" if chat_history else ""
    final_prompt = (
        f"Answer the question using the context below. If the context includes "
        f"both document and web sources, you may draw on both. Do not include "
        f"any citation markers, footnotes, or reference numbers in your answer "
        f"— write plain prose only.\n\n"
        f"{history_block}"
        f"Question: {question}\n\nContext:\n{combined_context}"
    )
    completion = _client.chat.completions.create(
        model=ANSWER_MODEL_NAME,
        messages=[{"role": "user", "content": final_prompt}],
    )

    return {
        "answer": completion.choices[0].message.content,
        "used_web_search": needs_web,
        "router_reason": reason,
    }


# ---------------------------------------------------------------------------
# Standalone tests
# ---------------------------------------------------------------------------
def _run_case(label: str, question: str, pdf_context: str):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    result = answer_question(question=question, pdf_context=pdf_context)
    print("Used web search:", result["used_web_search"])
    print("Router reason:", result["router_reason"])
    print("\nAnswer:\n", result["answer"])


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # picks up GROQ_API_KEY from your .env

    # Case 1: question your PDFs likely can't answer (needs live info)
    _run_case(
        label="CASE 1 — should trigger web search",
        question="What is the latest AI model released this week?",
        pdf_context="This document discusses machine learning fundamentals from 2023.",
    )

    # Case 2: question that SHOULD be answerable purely from PDF context.
    # Replace pdf_context below with a real chunk copied from one of your
    # actual DocMind PDFs, and phrase the question to match it directly.
    _run_case(
        label="CASE 2 — should NOT trigger web search",
        question="What is gradient descent used for according to this document?",
        pdf_context=(
            "Gradient descent is an optimization algorithm used to minimize "
            "a loss function by iteratively moving in the direction of "
            "steepest descent, as defined by the negative of the gradient. "
            "It is widely used to train machine learning models, including "
            "linear regression and neural networks."
        ),
    )