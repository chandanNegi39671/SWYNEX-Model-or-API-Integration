"""CLI:  python -m app.cli "my card has not arrived yet"   (no server needed)"""
import json, sys
from .classifier import IntentRouter
from .llm import ReplyDrafter
from .privacy import redact

def main(argv=None):
    texts = (argv if argv is not None else sys.argv[1:]) or ["my card has not arrived yet"]
    router, drafter = IntentRouter(), ReplyDrafter()
    for t in texts:
        clean, pii = redact(t); res = router.route([clean])[0]
        print(json.dumps({"input": clean, "pii_redacted": pii, **res, "reply": drafter.draft(clean, res)}, indent=2))

if __name__ == "__main__":
    main()
