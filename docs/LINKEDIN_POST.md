Completed Task 2 of my internship with @SWYNEX Technologies! 🚀

Task: integrate a model / public AI API into a small prototype.

What I built: a Banking Query Router API. A customer message goes in, and the service returns the intent (77 classes), a calibrated confidence score, an AUTO_ROUTE / HUMAN_REVIEW decision, and a drafted reply.

How it works:
🔹 My Task 1 scikit-learn model (90.9% accuracy) wrapped as a FastAPI REST service + browser UI + CLI
🔹 Google Gemini API integration for drafting replies, optional, with an offline template fallback so it never breaks
🔹 PII redaction (Luhn-validated card numbers, IBAN, email, phone) BEFORE anything is classified, logged or sent to an external API
🔹 API key only via environment variable, with an automated test that scans the repo for leaked secrets
🔹 21 automated tests, ~3 ms median latency on CPU

Biggest lesson: the LLM should write text, not make the routing decision. The calibrated model stays in control, and the LLM is just an optional, replaceable component.

I also documented what I verified live and what I didn't, because honest limits matter more than a perfect-sounding demo.

Repo: [paste GitHub link]

#SWYNEX #Internship #AI #MachineLearning #FastAPI #LLM #Python
