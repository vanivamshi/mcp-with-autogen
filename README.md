# MCP with AutoGen — AI agents with Notion

User → (ngrok) → Flask `/run` → AutoGen `RoundRobinGroupChat` → GPT-4o → MCP tools (Notion MCP + Time MCP over stdio) → Notion.

## Setup

Requires Python 3.10–3.12, Node.js (for `npx`), and `uv` (for `uvx mcp-server-time`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY and NOTION_API_KEY
```

In Notion, share the pages you want the agent to access with your integration.

## Run

```bash
python agent.py "List the pages in my workspace"   # CLI test
python app.py                                      # REST API on :7001 (+ ngrok if NGROK_AUTHTOKEN set)
```

```bash
curl -X POST http://localhost:7001/run -H "Content-Type: application/json" \
  -d '{"task": "Create a page called Meeting Notes with today’s date as the heading"}'
```

## AWS EC2 (optional)

On an Ubuntu instance: install Python, Node.js and uv, clone the project, create `.env`, then
`python app.py` (or `gunicorn -b 0.0.0.0:7001 app:app`) and open port 7001 in the security group.
