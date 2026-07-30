# Data-Analyst Telegram Bot

An LLM agent (OpenAI) that answers data-analysis questions sent over Telegram
and replies with **exactly one JSON object** in the shape the question asks for,
plus a public `log_url` to its run log — the contract used by the grader in
[`Jivraj-18/tds-p1-t2-2026-telegram-bot`](https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot).

```
Telegram msg ──► app.py ──► Agent (OpenAI + tools) ──► exact-shape JSON reply
                   │                │                        ▲
                   │                ├─ run_python  (internet + pandas/…)
                   │                └─ web_search  (DuckDuckGo)
                   └─ writes logs/<run_id>.jsonl, served at
                      PUBLIC_BASE_URL/logs/<run_id>.jsonl  ◄── that's log_url
```

## How it meets the grading contract

The grader (`collect.py` + `grade.py`) does exactly this per question:
sends each message and waits for **one reply per message**; for the last
message it runs `json.loads(reply.strip())` and compares to the expected value.
So this bot:

- **Replies once per message.** Multi-turn "setup" messages (e.g. *"build a
  model to forecast X"*) get a short `Ready.` ack; the message that actually
  asks for an answer gets the JSON.
- **Emits pure JSON, nothing else** for answer turns — no prose, no code
  fences — so `json.loads` succeeds.
- **Reproduces the exact requested shape.** If the message shows
  `{"state": "<state name>"}` it replies `{"state": "Assam"}`; if it shows
  `{"answer": {...}, "log_url": "<url>"}` it replies with that wrapper. The bot
  never force-wraps, so it matches whichever shape a question uses.
- **Fills `log_url`** with a real, wget-able URL whenever the question asks for
  one (the model only ever writes a placeholder; the server substitutes the
  true URL).
- **Grounds answers in real data** via `run_python` (full internet + pandas,
  numpy, requests, BeautifulSoup, openpyxl, pdfplumber, …) so it can download
  MOSPI/other public datasets and compute, rather than guess.

## Files

| file | purpose |
|------|---------|
| `app.py` | FastAPI service: Telegram transport, log hosting, orchestration |
| `agent.py` | OpenAI function-calling loop + system prompt |
| `tools.py` | `run_python` + `web_search` schemas and executors |
| `sandbox.py` | subprocess Python runner (timeout, mem limits, scrubbed env) |
| `answer.py` | JSON extraction + `log_url` injection + shape logic |
| `runlog.py` | per-run JSONL logger → `log_url` |
| `telegram_api.py` | tiny Telegram Bot API client |
| `config.py` | all env configuration |
| `run_local.py` | offline test (no Telegram) |

## 1. Create the bot

Message **@BotFather** → `/newbot` → copy the token into `TELEGRAM_BOT_TOKEN`.

## 2. Configure

```bash
cp .env.example .env
# fill OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, and PUBLIC_BASE_URL
```

`PUBLIC_BASE_URL` is the externally reachable URL of **this** service (no
trailing slash), e.g. `https://my-bot.fly.dev`. It's what `log_url` is built
from, so the grader can download the log — set it correctly.

## 3. Test locally (optional, no Telegram)

```bash
pip install -r requirements.txt
OPENAI_API_KEY=sk-... python run_local.py
OPENAI_API_KEY=sk-... python run_local.py 'Mean of [2,4,6]? Reply with ONLY {"mean": <n>}'
```

It prints the exact reply string and the run-log path.

## 4. Deploy (must stay ALWAYS-ON)

The grader may message at any time over hours/days, and downloads `log_url`
afterward — so the service must stay running and keep serving `/logs`.
Polling mode (default) works anywhere with outbound internet.

**Fly.io** (always-on + volume):
```bash
fly launch --no-deploy          # set a unique app name in fly.toml
fly volume create logs -s 1
fly secrets set OPENAI_API_KEY=sk-... TELEGRAM_BOT_TOKEN=... \
                PUBLIC_BASE_URL=https://<app>.fly.dev
fly deploy
```

**Render** (use a non-sleeping plan; blueprint in `render.yaml`): create a
Blueprint from the repo, set the three secret env vars, and set
`PUBLIC_BASE_URL` to your `https://<name>.onrender.com`.

**Docker anywhere / VPS:**
```bash
docker build -t da-bot .
docker run -d --restart=always -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... -e TELEGRAM_BOT_TOKEN=... \
  -e PUBLIC_BASE_URL=https://your.domain -v $PWD/logs:/app/logs da-bot
```

Verify: open `PUBLIC_BASE_URL/healthz` (should say `ok`), then DM your bot a
question and confirm the reply plus that `log_url` downloads.

### Polling vs webhook
- `BOT_MODE=polling` (default): the bot long-polls Telegram — no inbound
  webhook needed; survives platforms that would otherwise idle.
- `BOT_MODE=webhook`: set `WEBHOOK_SECRET`; the app registers
  `PUBLIC_BASE_URL/webhook/<secret>` on startup.

## 5. Try the real grader against your bot

```bash
git clone https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot
cd tds-p1-t2-2026-telegram-bot && pip install -r requirements.txt
# fill its .env (TELEGRAM_API_ID/HASH/SESSION via login.py), then a 1-row roster:
#   email,github_url,telegram_bot_username
#   you@x.edu,https://github.com/you/repo,@your_bot
python3 generate.py --students students.csv
python3 collect.py  --students students.csv
python3 grade.py    --students students.csv
```

Add your own questions to `evals/questions.json` to test coverage.

## Configuration reference

See `.env.example`. Key knobs: `OPENAI_MODEL` (default `gpt-4o`),
`AGENT_DEADLINE_SECONDS` (keep below the grader's `timeout_seconds`, default
300), `MAX_AGENT_STEPS`, `PYTHON_EXEC_TIMEOUT`, `HISTORY_IDLE_RESET_SECONDS`.

## Notes / caveats

- `run_python` executes model-generated code as a subprocess with a scrubbed
  env (no secrets), a timeout, and best-effort memory limits. It is a
  blast-radius reducer, not a hardened jail — fine for your own grading bot.
- Logs persist on disk; mount a volume (Fly/Render/Docker examples do) so they
  survive restarts and stay downloadable for later review.
- One worker only: chat state and the polling offset live in-process.
