# 90-second demo video script (screen recording)

**0:00-0:10** Show the GitHub repo name and README title. Say: "Task 2, I integrated my Task 1 model and the Google Gemini API into a working prototype."

**0:10-0:25** Terminal: `uvicorn app.main:app --reload`. Open http://127.0.0.1:8000/. Say: "This is a REST API with a small UI. It runs fully offline, no API key needed."

**0:25-0:45** Click the example chip `my card has not arrived yet` -> Route it. Point at: AUTO ROUTE badge, top-3 intents with confidence, drafted reply, latency in ms.

**0:45-1:00** Click the chip with the card number + email. Point at "PII redacted: CARD_NUMBER, EMAIL". Say: "Sensitive data is removed before the model, the logs or any external API ever sees it."

**1:00-1:10** Click `what's the weather like tomorrow` -> HUMAN REVIEW. Say: "Low confidence goes to a human instead of being routed wrongly."

**1:10-1:25** Open /docs (Swagger) -> try POST /v1/route. Then terminal: `pytest -q` showing 21 passed.

**1:25-1:30** Say: "The LLM is optional; the key lives only in an environment variable, never in the repo. Code and docs are on GitHub."

Optional bonus (only if you have a key): set `GEMINI_API_KEY`, restart, show `"mode": "gemini"` in the reply. Without a key, show the deterministic `"mode": "template"` fallback.
