# Anyone

**Simulate anyone from public sources. Every quote is real.**

Anyone builds an AI simulation of a person (a public figure, a private professional you're about to meet, or yourself) from text that person actually published. Most "AI clone" tools generate in someone's voice. Anyone only shows their words when it can prove they said them:

- **Verbatim quotes** are shown as the person's words only after Anyone checks that the exact words (ignoring only typographic differences like curly quotes) appear in a source Anyone fetched itself.
- **Extrapolation** is labeled as inference, written in the third person, and must cite the passages it relies on.
- **No evidence** is a first-class answer. If the person never said anything on the topic, the simulation says so instead of guessing.

The model *proposes*; the validator *decides*. A quote that doesn't match the source word for word is relabeled as extrapolation before anyone sees it. A citation the model invented is stripped. An inference with no supporting source is removed.

> Status: early alpha (v0.1.0.dev0), renamed from CiteSelf. Works locally with Claude. APIs and file formats may change.

## Why I built this

I read a newsletter, and I honestly can't remember whose, where someone described building an AI simulation of their boss. They wanted to rehearse before submitting a proposal. It worked up to a point: the answers sounded reasonable. But they didn't capture how the boss would actually respond, because the AI only carried what was in the data. It didn't know their mood that day, their state of mind, or what else was weighing on them.

That stuck with me. Preparing for a conversation with a specific person is useful: a manager, a client, an investor, a customer you're trying to understand. But a simulation that fills the gaps with confident guesses is worse than none, because you walk in prepared for the wrong person.

So Anyone is built around the gap. It only quotes what the person actually said, labels everything else as inference, and says "no evidence" when the record is silent. It can't tell you someone's mood. It can tell you exactly where its knowledge ends.

I'm also building it to test my own thinking about AI products: where the model should be trusted, where code should decide, and how you'd know if it's working.

— Yusuf Muhammed ([@yusstech](https://github.com/yusstech))

## Who you can simulate

Every persona has a `subject.kind` in `persona.yaml`:

| Kind | Built from | Guardrails |
|---|---|---|
| `public_figure` | Public sources by and about the person | Every answer says the person has not endorsed it. |
| `private_individual` | Public **professional** material only (work, career, published views) | Also says the person has not consented. Health, family, religion, sexuality and location are always off limits. |
| `self` | Your own writing ("verified self" mode) | You self-attest in a `consent` block. You may relax redaction for your own material. |

For every kind:

- **The disclosure is on every answer:** "AI simulation of X based on public sources; X has not endorsed this."
- **Personal data is redacted when material is added.** Email addresses, phone numbers and street addresses are replaced before anything is stored, indexed or sent to a model. This can't be turned off for `public_figure` or `private_individual`.
- **Your index stays on your machine.** It lives in one SQLite file next to `persona.yaml`. Anyone doesn't save conversations, but with a model provider such as Anthropic, each question and the retrieved passages are sent to that provider's API. Research (`anyone new`, `anyone deepen`) also sends the name, your hint and your links to the provider's web search.

Anyone is text-only. It does not clone voices or faces, and it does not format output to look like real posts.

## Quick start

Requires Python 3.10+.

```bash
pip install "anyone-ai[pdf,mcp] @ git+https://github.com/yusstech/PersonaLab"

# Someone else: Anyone researches them (needs an API key, below)
anyone new "Paul Graham" --link paulgraham.com --x-handle paulg --hint "essayist, Y Combinator co-founder"
anyone -C paul-graham ask "Should founders do things that don't scale?"

# Yourself
anyone -C me init --name "Your Name"                     # kind: self; asks you to self-attest
anyone -C me add ~/writing/                              # files/folders: .md .txt .html .pdf .docx
```

For real answers, Anyone uses Claude with **your own API key**. Keys are read from the environment and never written to disk, logs or `persona.yaml`:

```bash
export ANTHROPIC_API_KEY=...               # or, on macOS, keep it in Keychain:
# security add-generic-password -a "$USER" -s anyone-anthropic -w
# export ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s anyone-anthropic -w)"

anyone -C paul-graham chat                 # multi-turn, in the terminal
anyone -C paul-graham serve                # web chat with a source panel at http://127.0.0.1:8765
anyone -C paul-graham ask "..." --provider extractive   # offline: closest verbatim passages, no API key
```

Example answer:

```
AI simulation of Avery Moreno based on public sources; Avery Moreno has not endorsed this. This is not Avery Moreno. ...

  "Release quality depends on the severity of defects and whether we have a safe rollback."  [E1 ✓ verbatim]
  ~ Based on these passages, Avery Moreno would likely accept a known minor bug in a small,
    controlled release if a rollback exists.  [extrapolation from E1]
  ∅ The evidence does not say what counts as "minor" for them.

  Sources:
   E1  Talk on shipping and quality (2026-03-21)  · src-8f945b0577
```

## Research someone

`anyone new "<name>"` builds a persona from public sources in four steps:

1. **Discover.** Claude's built-in web search looks for their own writing, interviews, talks and X posts. A URL is kept only if a real search result contained it, so the model can't invent sources.
2. **Collect.** Anyone fetches every text itself and never uses a model's summary as quotable text. X posts come from X's public oEmbed endpoint, which returns the exact post, author and date. Pages come from a robots.txt-respecting fetcher. Archive pages on a site you linked are followed, up to 50 pages a run. People-search sites and private network addresses are never fetched.
3. **Classify.** This decides what counts as the person's own words, and the model's opinion alone never does:
   - **X posts** count only when oEmbed names a *confirmed* account as the author. Accounts are confirmed by your `--x-handle`, your x.com link, or your answer when the model proposes one. Posts by anyone else are skipped.
   - **Pages on a site you linked** count as theirs.
   - **Every other page** is `secondhand` reporting.
4. **Review.** You see the dossier, grouped into their own words and reporting about them. Remove anything about the wrong person before you chat.

| Option | Use |
|---|---|
| `--link URL` | Their site, x.com profile, a talk page (repeatable). The strongest signal of identity. |
| `--hint TEXT` | Who they are, to tell namesakes apart: `"CEO of Acme, Lagos"` |
| `--x-handle H` | Their X account (repeatable) |
| `--kind private_individual` | Professional material only. The search is told so and people-search sites are blocked. |
| `--max-searches N` | Default 8. Web search costs $10 per 1,000 searches, plus model tokens. A run is typically well under $1. |

`anyone deepen "<topic>"` runs another targeted search for an existing persona. Re-running never stores the same text twice.

## Use it from any AI agent (MCP)

`anyone mcp` serves a persona over the [Model Context Protocol](https://modelcontextprotocol.io) with three tools:

| Tool | What it does |
|---|---|
| `ask_persona` | Cited, labeled answer (quote / extrapolation / no_evidence) |
| `search_record` | Verbatim passages from the sources. No AI generation. |
| `persona_info` | Disclosure text, subject kind, sources and boundaries |

Claude Code, with the key read from macOS Keychain at launch (never written to config):

```bash
claude mcp add anyone -- sh -c 'ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s anyone-anthropic -w)" exec anyone -C ~/dana mcp'
```

## persona.yaml

```yaml
anyone: 2
subject:
  kind: public_figure            # public_figure | private_individual | self
  name: Dana Reyes
  bio: One line shown to visitors.
research:                        # written by `anyone new`; used again by `anyone deepen`
  links: [https://danareyes.com]        # pages on these hosts count as their own words
  hint: CEO of Acme, Lagos
  x_handles: [danareyes]                # confirmed accounts: their posts count as their own words
# consent:                       # only for kind: self (your self-attestation); rejected for other kinds
boundaries:
  blocked_keywords: [salary]            # questions containing these are refused before any model call
  blocked_topics: [politics]            # model is told to decline these (private_individual adds five more)
  allow_extrapolation: true             # false = quotes and "no evidence" only
privacy:
  redact_names: [Jordan Lee]            # other people's names, replaced when material is added
  redact_pii: true                      # emails, phones, street addresses; only kind self may set false
model:
  provider: anthropic                   # or: extractive (offline, quotes only)
  name: claude-opus-5-5
  effort: medium
  refusal_fallback: true
retrieval:
  top_k: 6
  min_quote_words: 3
```

CiteSelf manifests (`citeself: 1`) still load, as `kind: self`, and a `.citeself/` index is moved to `.anyone/` the first time it is used.

## Adding sources

`anyone add <anything>` works out what you gave it:

| You give it | What happens |
|---|---|
| A website or blog URL | Finds the RSS/Atom feed, the WordPress API or the sitemap, then imports the main text and date of each page. Follows robots.txt (RFC 9309), stays on that domain, makes about 1 request per second, and skips privacy, login and tag pages. Re-run it to pick up new or edited posts. Use `--dry-run` to preview. |
| A file or folder | `.md` `.txt` `.html` `.pdf` `.docx` |

**Authorship decides what can be quoted.** Each import is marked:

- `by_me`: written by the subject, word for word (you, in self mode). Can be quoted as their words.
- `secondhand`: reporting about them by an outlet. A quote from it is verified only if the outlet itself puts those words in quotation marks *and* names the subject in the same attribution clause (`"...," Reyes said.`). It is shown as **"as reported by <outlet>"**. The outlet's own sentences, and remarks it attributes to other people, are never shown as the subject's words.
- `ai_assisted`: ghostwritten or AI-drafted text. Used only as context and **never shown as their verbatim words**, even if the model tries.

For other people, Anyone never assumes authorship. Pass `--authorship` or answer the prompt. Change it later with `anyone authorship <source-id> <by_me|secondhand|ai_assisted>`.

## Managing sources

```bash
anyone sources                 # list what's ingested
anyone exclude src-1a2b3c4d5e  # stop using a source in answers
anyone include src-1a2b3c4d5e
anyone remove  src-1a2b3c4d5e  # delete it from the index
anyone revoke                  # delete the whole index and write a revocation receipt
```

Re-adding an edited file replaces the old version. Everything Anyone derives lives in `.anyone/` next to `persona.yaml`; original files are never modified.

## Measure fidelity (self mode)

A convincing answer isn't the same as a faithful one. In self mode you can test the simulation against your own answers:

```bash
anyone eval init     # creates eval.yaml: write questions AND your own answers, before looking at any AI output
anyone eval run      # asks every question three ways, then judges each answer against yours
anyone eval score .anyone/eval/<run>   # after you blind-rate rating_sheet.csv yourself
```

Three arms answer each question: **anyone** (the full pipeline), **bio_only** (a generic model given the name and bio) and **generic** (name only). The report shows:

- **Alignment** with your own answers (1–5, scored by a judge model that isn't told which arm wrote what) and **contradiction rate**
- **Normalized score:** if you re-answer questions a week later (`my_answer_retest`), your own consistency becomes the ceiling, following [Park et al. 2024](https://arxiv.org/abs/2411.10109)
- **Grounding:** share of proposed quotes that were verbatim, and whether it says "no evidence" on questions your material doesn't cover (and only those)
- A **blind rating sheet** with shuffled, unlabeled answers for you to score. Your ratings are the strongest signal; the judge is an LLM and can be wrong.

Write answers in your own words rather than copying from your material, and include questions your writing doesn't cover. Otherwise the test is circular. A backtest for public figures (hide a real reply, simulate it, compare) is on the roadmap.

## How it works

1. **Ingest:** text is extracted, cleaned, redacted and split into chunks that keep their exact offsets in the source. It is stored in one local SQLite file with full-text search.
2. **Retrieve:** the question (plus the previous question, for follow-ups) is matched against the chunks with BM25. If nothing matches, the model is not called.
3. **Generate:** Claude receives only the retrieved passages, wrapped as data, and returns structured JSON segments.
4. **Validate:** each segment is checked against the passages it cites (see `src/anyone_ai/validate.py`). What fails is relabeled or removed, and every decision is reported in the answer's `validation` block.

## Honest limits

- **Verbatim ≠ in context.** Anyone proves a quote exists in the source. It cannot prove the quote is being used the way the person meant it.
- **Extrapolation is inference.** It is labeled and cited, but whether it's a *fair* reading is a judgment call.
- **Redaction favours precision.** It catches common email, phone and street-address formats. It can miss addresses written house-number-last, obfuscated emails and unformatted digit runs. Review what you add.
- **Retrieval is keyword-based** in v0.1, so it can miss passages that use different words.
- **Research can pick up the wrong person** with a common name. Links and a hint help; the dossier review is where you catch it.
- **Secondhand attribution is a heuristic.** Quotes attributed only by pronoun ("she said") are not shown as quotes, and a misattribution in the article itself would carry through. That is why such quotes are labeled with their outlet.
- **Deleted X posts are not yet re-checked.** X's terms require dropping posts that are deleted or protected. Until `anyone sync` exists, re-run research or remove them by hand.
- **Self-attestation is not identity verification.** For `kind: self`, `persona.yaml` records your consent; it can't prove you are who you say.
- **`anyone serve` has no authentication.** It binds to `127.0.0.1` by default. Don't expose it to a network as is.
- **You are responsible for how you use it.** Laws on AI likenesses, defamation, data protection and AI disclosure (e.g. EU AI Act Art. 50, GDPR) vary by place. Anyone discloses on every answer and limits what it collects; it can't make every use lawful.

## Roadmap

1. **More models:** Grok (xAI) and OpenAI adapters, chosen per persona. Bring your own keys (`XAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
2. **Reply simulator:** `anyone reply "<post>"` gives a labeled, simulated reply, an engagement estimate and the verified quotes behind it.
3. **Public bring-your-own-key web mode:** visitors use their own key, held in memory for the session only, and a web dossier review.
4. **Backtest evaluation** for public figures.
5. **`anyone sync`:** re-check stored X posts and drop deleted ones.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                  # offline test suite
.venv/bin/pytest -m live          # also calls the real API (needs ANTHROPIC_API_KEY)
```

The test persona, Avery Moreno, is entirely fictional.

## How this was built

I designed it and made the product and trust decisions. The code was written with Claude (Anthropic) as an AI pair programmer under my direction, and the commits are co-authored. The offline test suite covers every change.

Questions and feedback: [GitHub Issues](https://github.com/yusstech/PersonaLab/issues) or [@yusstech](https://github.com/yusstech).

## License

Apache-2.0 © 2026 Yusuf Muhammed
