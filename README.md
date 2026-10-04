# Support Ticket Triage Agent

A small LLM agent that triages support tickets for a **fictional** train-station assistance service. For each ticket it searches a knowledge base, drafts a reply, picks a category and priority, and decides whether a human must handle it.

All data is synthetic: the service, the knowledge base and the 30 test tickets were written by me.

## What it does

Given a ticket such as *"Can I change my booking to a later time?"*, the agent returns:

```json
{
  "category": "booking",
  "priority": "medium",
  "reply": "Yes, you can change your booking in the app up to 2 hours before the assistance time...",
  "escalate": false
}
```

Rules built into the agent:
- It only states facts found in the knowledge base. If the answer is not there, it escalates.
- It never promises refunds or compensation.
- Safety problems, injuries, discrimination, legal threats, service failures and requests for money are escalated to a human.

## How it works

1. The ticket and two tools are sent to an LLM through an OpenAI-compatible API.
2. The model calls `search_kb`, which searches `kb.md` by keyword. This runs locally.
3. The model calls `submit_triage` with its final structured answer.
4. If the model does not finish within 6 steps, the ticket is escalated by default.

The code also retries when the free API tier rate-limits a request, and asks the model to try again if it answers in plain text or sends invalid JSON.

Run on the free tier of Groq with `openai/gpt-oss-20b`. Any OpenAI-compatible provider with tool calling should work (set `PROVIDER`, `MODEL` and `LLM_API_KEY`).

## Results

I labelled 30 tickets by hand (category and whether to escalate) and measured how often the agent agrees with my labels.

| | First prompt (1 run) | Improved prompt (5 runs) |
|---|---|---|
| Category accuracy | 28/30 (93%) | 28 to 30 of 30 (93 to 100%) |
| Escalation accuracy | 20/30 (67%) | 27 to 28 of 30 (90 to 93%) |

Results vary slightly from run to run even with the temperature set to 0, so I report a range for the improved prompt.

**What I changed:** the first prompt answered many tickets that should have gone to a human (late staff, missed callbacks, requests for money back, questions the knowledge base cannot answer). I added explicit escalation rules for these cases, a rule for tickets that mix topics, and a rule for lost items.

**Remaining errors:**
- Ticket #7 (how far ahead to book) was escalated in all 5 runs, even though the knowledge base answers it. It was correct with the first prompt, so the new rules caused this regression.
- Ticket #13 (a request for guiding a blind passenger, which is not in the knowledge base) was answered in all 5 runs instead of being escalated. It was wrong with the first prompt too.
- Tickets #26 and #30 failed in some runs only: an angry customer who only wants to change a booking, and a question about cancellation charges that was sometimes filed under booking.

**Caveat and held-out test:** I improved the prompt using the same 30 tickets I measure on, so those scores are optimistic. To check this, I ran the agent once, without changing anything, on 10 tickets I had not used before (`tickets_unseen.json`):

| | Unseen tickets (1 run) |
|---|---|
| Category accuracy | 8/10 (80%) |
| Escalation accuracy | 10/10 (100%) |

Both misses were category mix-ups, not escalation errors: a question about cancelling (#4) was filed as booking, and a luggage question (#8) was filed as accessibility request instead of other. Category accuracy is lower here than on the 30 tickets I tuned on, which suggests those scores were optimistic. Ten tickets is a very small sample, so I would not draw firm conclusions from either number.

## Limitations

- Keyword search over a 7-entry knowledge base. A real system would need better retrieval.
- 30 test tickets is a small sample, and the labels are my own judgement.
- Free-tier models are less reliable at tool calling than larger paid ones, and results can vary slightly between runs.

## Run it

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

export LLM_API_KEY="your-key"
export PROVIDER="groq"
export MODEL="openai/gpt-oss-20b"

python agent.py "Can I change my booking to a later time?"
python agent.py eval            # scores the agent on tickets.json
python agent.py eval tickets_unseen.json   # scores it on the 10 held-out tickets
```

Never commit your API key.

## Files

- `agent.py`: the agent loop, tools and evaluation
- `kb.md`: the knowledge base
- `tickets.json`: 30 labelled test tickets
- `tickets_unseen.json`: 10 labelled tickets kept aside for a final check

## Next steps

- Replace keyword search with embeddings.
- Add a small web interface and results dashboard.