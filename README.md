# CiteSelf

**An AI of you that only speaks on the record.**

CiteSelf turns your own writing (essays, newsletters, talk transcripts, notes) into an AI that people can consult about your views. Unlike other "AI clone" tools, it never puts words in your mouth:

- **Verbatim quotes** are shown as your words only after CiteSelf checks, character by character, that you actually wrote them in the cited source.
- **Extrapolation** is labeled as inference, written in the third person, and must cite the passages it relies on.
- **No evidence** is a first-class answer. If you never wrote about it, the simulation says so instead of guessing.

The model *proposes*; the validator *decides*. A quote that doesn't match the source word for word is relabeled as extrapolation before anyone sees it. A citation the model invented is stripped. An inference with no supporting source is removed.

> Status: early alpha (v0.1.0.dev0). Works locally with Claude. APIs and file formats may change.

## Who it's for

Creators and experts with a body of work — writers, podcasters, coaches, researchers, educators — who want their audience to be able to ask "what would they say about X?" without risking an AI confidently misquoting them.

**CiteSelf only simulates yourself.** There is no mode for simulating other people, deceased people, or anyone who hasn't attested to it. It is text-only: no voice or face cloning.

## Quick start

Requires Python 3.10+.

```bash
pip install "citeself[pdf,mcp] @ git+https://github.com/yusstech/PersonaLab"

mkdir me && cd me
citeself init --name "Your Name"          # creates persona.yaml and asks you to self-attest
citeself ingest ~/writing/                 # .md .txt .html .pdf files or folders
citeself ask "How do you think about pricing?" --provider extractive   # offline, no API key needed
```

For real answers, CiteSelf uses Claude. Provide an Anthropic API key via the environment:

```bash
export ANTHROPIC_API_KEY=...               # or, on macOS, keep it in Keychain:
# security add-generic-password -a "$USER" -s citeself-anthropic -w
# export ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s citeself-anthropic -w)"

citeself ask "How do you think about pricing?"
citeself chat                              # multi-turn, in the terminal
citeself serve                             # web chat with a source panel at http://127.0.0.1:8765
```

Example answer:

```
AI simulation of Avery Moreno, built from Avery Moreno's own published material. This is not Avery Moreno. ...

  "Release quality depends on the severity of defects and whether we have a safe rollback."  [E1 ✓ verbatim]
  ~ Based on what Avery Moreno has written, they would likely accept a known minor bug in a small,
    controlled release if a rollback exists.  [extrapolation from E1]
  ∅ The evidence does not say what counts as "minor" for them.

  Sources:
   E1  Talk on shipping and quality (2026-03-21)  · src-8f945b0577
```

## Use it from any AI agent (MCP)

`citeself mcp` serves your persona over the [Model Context Protocol](https://modelcontextprotocol.io) with three tools:

| Tool | What it does |
|---|---|
| `ask_persona` | Cited, labeled answer (quote / extrapolation / no_evidence) |
| `search_record` | Verbatim passages from your material. No AI generation. |
| `persona_info` | Disclosure text, sources and your boundaries |

Claude Code, with the key read from macOS Keychain at launch (never written to config):

```bash
claude mcp add citeself -- sh -c 'ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s citeself-anthropic -w)" exec citeself -C ~/me mcp'
```

## persona.yaml

```yaml
citeself: 1
persona:
  name: Your Name
  bio: One line shown to visitors.
consent:                       # required: you attest this is you
  self_attested: true
  attested_by: Your Name
  attested_at: "2026-11-04"
  statement: I am Your Name. I am creating an AI simulation of myself ...
boundaries:
  blocked_keywords: [salary, my kids]   # hard block: never sent to the model
  blocked_topics: [health, family]      # model is told to decline these
  allow_extrapolation: true             # false = quotes and "no evidence" only
privacy:
  redact_names: [Jordan Lee]            # other people's names, replaced at ingest
model:
  provider: anthropic                   # or: extractive (offline, quotes only)
  name: claude-opus-5-5
  effort: medium
  refusal_fallback: true
retrieval:
  top_k: 6
  min_quote_words: 3
```

## Managing your corpus

```bash
citeself sources                 # list what's ingested
citeself exclude src-1a2b3c4d5e  # stop using a source in answers
citeself include src-1a2b3c4d5e
citeself remove  src-1a2b3c4d5e  # delete it from the index
citeself revoke                  # delete the whole index and write a revocation receipt
```

Re-ingesting an edited file replaces the old version. Everything CiteSelf derives lives in `.citeself/` next to `persona.yaml`; your original files are never modified. Conversations are not stored anywhere.

## How it works

1. **Ingest:** files are extracted, cleaned, redacted and split into chunks that keep their exact offsets in the source. They are stored in one local SQLite file with full-text search.
2. **Retrieve:** the question (plus the previous question, for follow-ups) is matched against your chunks with BM25. If nothing matches, the model is not called.
3. **Generate:** Claude receives only the retrieved passages, wrapped as data, and returns structured JSON segments.
4. **Validate:** each segment is checked against the passages it cites (see `src/citeself/validate.py`). What fails is relabeled or removed, and every decision is reported in the answer's `validation` block.

## Honest limits

- **Verbatim ≠ in context.** CiteSelf proves a quote exists in your material. It cannot prove the quote is being used the way you meant it.
- **Extrapolation is inference.** It is labeled and cited, but whether it's a *fair* reading of your views is a judgment call. A fidelity evaluation (`citeself eval`, coming next) will measure this against your own answers.
- **Retrieval is keyword-based** in v0.1, so it can miss passages that use different words. Embedding search is on the roadmap.
- **Self-attestation is not identity verification.** `persona.yaml` records your consent; it can't prove you are who you say. Anyone hosting CiteSelf publicly must add their own verification.
- **`citeself serve` has no authentication.** It binds to `127.0.0.1` by default. Don't expose it to a network as is.
- Laws on AI likenesses and AI disclosure (e.g. EU AI Act Art. 50) vary by place. CiteSelf shows an AI disclosure on every answer, but you are responsible for how you deploy it.

## Roadmap

- `citeself eval`: measure fidelity against your own held-out answers, compared with generic-AI and bio-only baselines
- Embedding and hybrid retrieval
- Export as an Agent Skill; Docker image for a hosted "ask me" page
- Track how your views change over time

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                  # offline test suite
.venv/bin/pytest -m live          # also calls the real API (needs ANTHROPIC_API_KEY)
```

The test persona, Avery Moreno, is entirely fictional.

## License

Apache-2.0
