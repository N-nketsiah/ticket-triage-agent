"""Support-ticket triage agent (demo with a fictional service desk).
Works with any OpenAI-compatible LLM API that supports tool calling (Groq, Gemini, Mistral, OpenRouter).

Setup:
  pip install openai
  export LLM_API_KEY="your-key"
  export PROVIDER="groq"            # groq | gemini | mistral | openrouter (default: groq)
  export MODEL="model-name"         # pick a current model with tool calling from the provider's docs

Usage:
  python agent.py "ticket text here"   # triage one ticket
  python agent.py eval                 # run all tickets in tickets.json and score them
  python agent.py eval other.json      # score a different ticket file
"""
import json
import os
import re
import sys
import time

PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
MODEL = os.environ.get("MODEL", "")
MAX_STEPS = 6
DELAY = float(os.environ.get("DELAY", "2"))  # seconds between tickets in eval, to respect rate limits

SYSTEM = """You are a support triage agent for StationAssist, a fictional service that books passenger assistance at train stations.
For each ticket: search the knowledge base for relevant answers, then call submit_triage exactly once.
Rules:
- Only state facts found in the knowledge base. If the knowledge base does not answer the question, do not guess: escalate.
- Never promise refunds or compensation.
- Escalate anything about safety, injuries, discrimination or legal threats.
- Escalate every complaint about the service failing (staff late, staff not arriving, promised calls not made) and every request for money back or compensation, even if the ticket also contains a simple request.
- If a ticket mixes topics, choose the category of the most serious issue: safety first, then complaint, then the others.
- Lost items are not escalated: reply that the train operator handles lost property.
- Keep replies short, polite and plain."""

CATEGORIES = ["booking", "cancellation", "accessibility_request", "complaint", "safety", "other"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the knowledge base. Returns up to 3 relevant entries.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keywords to search for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_triage",
            "description": "Submit the final triage result for the ticket. Call this once, at the end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "reply": {"type": "string", "description": "Draft reply to the customer"},
                    "escalate": {"type": "boolean", "description": "True if a human must handle this"},
                    "escalation_reason": {"type": "string"},
                },
                "required": ["category", "priority", "reply", "escalate"],
            },
        },
    },
]


def load_kb(path="kb.md"):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, path), encoding="utf-8") as f:
        text = f.read()
    return [part.strip() for part in text.split("## ") if part.strip()]


def search_kb(query, kb=None):
    kb = kb if kb is not None else load_kb()
    words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for entry in kb:
        score = len(words & set(re.findall(r"\w+", entry.lower())))
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: -pair[0])
    if not scored:
        return "No relevant entries found."
    return "\n\n".join(entry for _, entry in scored[:3])


def call_llm(client, messages):
    """One LLM call with a simple retry when the free tier rate limit (HTTP 429) is hit."""
    for attempt in range(4):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0
            )
        except Exception as e:
            if getattr(e, "status_code", None) == 429 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise


def run_agent(client, ticket_text):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Ticket:\n{ticket_text}"},
    ]
    for _ in range(MAX_STEPS):
        msg = call_llm(client, messages).choices[0].message
        if not msg.tool_calls:
            # Some free models answer in plain text. Remind them to use the tool.
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": "Call submit_triage now with your final answer."})
            continue
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            if args is None:
                output = "Your arguments were not valid JSON. Try again."
            elif tc.function.name == "submit_triage":
                if all(k in args for k in ("category", "priority", "reply", "escalate")):
                    return args
                output = "Missing fields. Include category, priority, reply and escalate."
            elif tc.function.name == "search_kb":
                output = search_kb(args.get("query", ""))
            else:
                output = "Unknown tool."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
    # Safe fallback: if the agent did not finish, hand the ticket to a human.
    return {"category": "other", "priority": "medium", "reply": "", "escalate": True,
            "escalation_reason": "Agent did not finish"}


def evaluate(client, filename="tickets.json"):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, filename), encoding="utf-8") as f:
        tickets = json.load(f)
    right_cat = right_esc = 0
    for t in tickets:
        result = run_agent(client, t["text"])
        ok_cat = result["category"] == t["category"]
        ok_esc = bool(result["escalate"]) == t["escalate"]
        right_cat += ok_cat
        right_esc += ok_esc
        if not (ok_cat and ok_esc):
            print(f"MISS #{t['id']}: got {result['category']}/escalate={result['escalate']}, "
                  f"expected {t['category']}/escalate={t['escalate']}")
        time.sleep(DELAY)
    n = len(tickets)
    print(f"Category accuracy: {right_cat}/{n} ({100 * right_cat / n:.0f}%)")
    print(f"Escalation accuracy: {right_esc}/{n} ({100 * right_esc / n:.0f}%)")


def make_client():
    from openai import OpenAI  # imported here so the file can be tested without the SDK
    provider = os.environ.get("PROVIDER", "groq")
    if provider not in PROVIDERS:
        sys.exit(f"Unknown PROVIDER. Choose one of: {', '.join(PROVIDERS)}")
    key = os.environ.get("LLM_API_KEY")
    if not key:
        sys.exit("Set the LLM_API_KEY environment variable first.")
    if not MODEL:
        sys.exit("Set the MODEL environment variable (see the provider's model list).")
    return OpenAI(base_url=PROVIDERS[provider], api_key=key)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    client = make_client()
    if sys.argv[1] == "eval":
        evaluate(client, sys.argv[2] if len(sys.argv) > 2 else "tickets.json")
    else:
        print(json.dumps(run_agent(client, " ".join(sys.argv[1:])), indent=2))


if __name__ == "__main__":
    main()