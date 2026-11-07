import pytest

from anyone_ai.manifest import DATA_DIR
from anyone_ai.redact import ADDRESS_TOKEN, EMAIL_TOKEN, PHONE_TOKEN, redact_names, redact_pii
from anyone_ai.store import Store

REDACTED = [
    ("Email me at jane.doe+pm@example.co.uk.", f"Email me at {EMAIL_TOKEN}."),
    ("Call +1 (415) 555-0132 today", f"Call {PHONE_TOKEN} today"),
    ("Call +44 20 7946 0958.", f"Call {PHONE_TOKEN}."),
    ("WhatsApp +2348031234567", f"WhatsApp {PHONE_TOKEN}"),
    ("(415) 555-0132", PHONE_TOKEN),
    ("415.555.0132", PHONE_TOKEN),
    ("1-800-555-0199", PHONE_TOKEN),
    ("ring 020 7946 0958", f"ring {PHONE_TOKEN}"),
    ("0803 123 4567", PHONE_TOKEN),
    ("(020) 7946-0958", PHONE_TOKEN),
    ("I live at 221B Baker Street, London", f"I live at {ADDRESS_TOKEN}, London"),
    ("office: 1600 Pennsylvania Avenue NW", f"office: {ADDRESS_TOKEN}"),
    ("12 Adeola Odeku Street, Victoria Island", f"{ADDRESS_TOKEN}, Victoria Island"),
    ("350 Fifth Ave, Suite 3400", ADDRESS_TOKEN),
    ("Write to PO Box 1234", f"Write to {ADDRESS_TOKEN}"),
]

# Text that must survive untouched: a false positive would corrupt words later quoted as verbatim.
UNCHANGED = [
    "On 2026-11-07 we shipped", "Revenue grew to 1,000,000 users", "server 192.168.100.200 was down",
    "v1.2.3 and 3.14159", "follow @elonmusk on X", "https://medium.com/@user/post", "In 2019 we hired 40 people",
    "ISBN 978-3-16-148410-0", "We grew from 3 to 12 teams", "Read St. Augustine", "scores 100 200 300",
    "met 3 Dr. Smith", "id 12345678901234567890", "a 10-12 hour day", "timestamp 1730966400", "+5 points",
    "about 20 people in 2024 at 9 am", "Q3 2026: 45% growth, 1.5x",
]


@pytest.mark.parametrize("text,expected", REDACTED)
def test_pii_redacted(text, expected):
    assert redact_pii(text) == expected


@pytest.mark.parametrize("text", UNCHANGED)
def test_ordinary_text_untouched(text):
    assert redact_pii(text) == text


def test_redact_names_longest_first_case_insensitive():
    assert redact_names("jordan lee and Jordan met", ["Jordan", "Jordan Lee"]) == "[redacted] and [redacted] met"


def test_ingest_redacts_before_storage_and_index(tmp_path):
    text = "Reach me at dana@example.com or +1 415 555 0132. Our office is 12 Market Street. Shipping beats polish."
    with Store(tmp_path / DATA_DIR / "index.db") as store:
        row = store.ingest_text("note", "Note", None, text)
        stored = store.conn.execute("SELECT text FROM sources WHERE id = ?", (row.id,)).fetchone()[0]
        assert "dana@example.com" not in stored and "555" not in stored and "Market Street" not in stored
        assert EMAIL_TOKEN in stored and PHONE_TOKEN in stored and ADDRESS_TOKEN in stored
        assert not store.search("example.com dana 0132 Market")  # nothing redacted is searchable
        assert store.search("shipping polish")

        # Self mode may opt out; the opted-out version replaces the redacted one under the same key.
        row2 = store.ingest_text("note", "Note", None, text, redact_pii=False)
        assert row2 and "dana@example.com" in store.conn.execute(
            "SELECT text FROM sources WHERE id = ?", (row2.id,)).fetchone()[0]
        assert store.stats()["sources"] == 1
