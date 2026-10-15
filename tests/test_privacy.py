from app.privacy import redact

def test_valid_card_redacted():
    t, f = redact("my card 4539 1488 0343 6467 was declined"); assert "[CARD_NUMBER]" in t and "4539" not in t and f == ["CARD_NUMBER"]

def test_non_luhn_number_kept():
    t, f = redact("order number 1234 5678 9012 3456"); assert "CARD_NUMBER" not in f

def test_email_phone_iban():
    t, f = redact("mail a.b@x.co call +44 7911 123456 iban GB82 WEST 1234 5698 7654 32")
    assert set(f) == {"EMAIL", "PHONE", "IBAN"} and "@" not in t and "7911" not in t and "WEST" not in t

def test_plain_text_untouched():
    s = "my card has not arrived after 2 weeks"; assert redact(s) == (s, [])
