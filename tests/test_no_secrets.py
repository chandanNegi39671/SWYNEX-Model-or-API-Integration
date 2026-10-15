"""Guard for the task rule 'do not include secret API keys': scan every tracked text file."""
import pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
PATTERNS = [re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"), re.compile(r"sk-[A-Za-z0-9]{20,}"), re.compile(r"AIza[0-9A-Za-z_-]{30,}"), re.compile(r"ghp_[A-Za-z0-9]{30,}"), re.compile(r"AQ\.[A-Za-z0-9_-]{30,}")]

def test_no_secrets_committed():
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".json", ".txt", ".html", ".example", ".csv", ".sh", ""} and ".venv" not in p.parts and "joblib" not in p.name and p.name != ".env":  # local .env is git-ignored (checked below)
            txt = p.read_text(errors="ignore")
            for pat in PATTERNS: assert not pat.search(txt), f"possible secret in {p}"

def test_env_file_is_gitignored():
    assert ".env" in (ROOT / ".gitignore").read_text().split()