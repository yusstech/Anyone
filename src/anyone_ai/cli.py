"""anyone command-line interface."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import __version__
from .engine import Answer, Conversation, Persona
from .governance import revoke
from .manifest import SUBJECT_KINDS, Manifest, ManifestError, find_root, index_path, load_manifest, write_new_manifest
from .providers import ProviderError
from .store import Store


class _Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text


style = _Style(sys.stdout.isatty() and not os.environ.get("NO_COLOR"))


def render(answer: Answer) -> str:
    out = [style("2", answer.disclosure), ""]
    for seg in answer.segments:
        refs = ", ".join(seg.evidence_ids)
        if seg.kind == "quote":
            out.append(style("1", f"  “{seg.text}”") + style("32", f"  [{refs} ✓ verbatim]"))
        elif seg.kind == "extrapolation":
            tag = f"  [extrapolation from {refs}]"
            out.append(f"  ~ {seg.text}" + style("33", tag))
            if seg.note:
                out.append(style("2", f"    ({seg.note})"))
        else:
            out.append(style("36", f"  ∅ {seg.text}"))
    cited = answer.cited_labels()
    if cited:
        out.append("")
        out.append(style("2", "  Sources:"))
        for label in cited:
            e = answer.evidence[label]
            out.append(style("2", f"   {label}  {e.title}" + (f" ({e.date})" if e.date else "") + f"  · {e.source_id}"))
    v = answer.validation
    if v.quotes_demoted or v.removed or v.stripped_citations:
        out.append(style("2", f"  Checks: {v.quotes_verified} quote(s) verified, {v.quotes_demoted} demoted, "
                              f"{len(v.removed)} segment(s) removed, {len(v.stripped_citations)} bad citation(s) stripped."))
    return "\n".join(out)


def _root(args) -> Path:
    return find_root(Path(args.dir))


_THIRD_PARTY_NOTICE = {
    "public_figure": "public sources",
    "private_individual": "public PROFESSIONAL material only (work, career, published views)",
}


def _confirm_init(args) -> bool:
    name, kind = args.name, args.kind
    if kind == "self":
        if args.yes:
            raise ValueError("--yes is for simulating someone else; for yourself use --i-am-this-person")
        if args.i_am_this_person:
            return True
        print("Self mode builds a simulation of YOURSELF, from material you wrote or have the right to use.")
        return input(f"Are you {name}, and do you agree to that? [y/N] ").strip().lower() in ("y", "yes")
    if args.i_am_this_person:
        raise ValueError("--i-am-this-person is only for --kind self")
    if args.yes:
        return True
    print(f"You are creating an AI simulation of {name}, who is not you, from {_THIRD_PARTY_NOTICE[kind]}.")
    print(f"  - {name} has not endorsed{' or consented to' if kind == 'private_individual' else ''} this; "
          "every answer says so.")
    if kind == "private_individual":
        print("  - Health, family, religion, sexuality and location are always off limits.")
    print("  - Emails, phone numbers and street addresses are redacted from everything you add.")
    print(f"  - Don't present output as {name}'s real words or posts. Only verified quotes are their words.")
    return input("Continue? [y/N] ").strip().lower() in ("y", "yes")


def cmd_init(args) -> int:
    root = Path(args.dir).resolve()
    if not _confirm_init(args):
        print("Aborted. Nothing was created.")
        return 1
    root.mkdir(parents=True, exist_ok=True)
    path = write_new_manifest(root, args.name, args.kind)
    hint = "anyone ingest <your-writing-folder>" if args.kind == "self" else \
        f"anyone add <url-or-folder>   (public material by {args.name})"
    print(f"Created {path}\nNext: {hint}")
    return 0


def _ask_authorship(args, what: str, manifest: Manifest) -> str:
    """Authorship decides whether text may be quoted as the subject's own words, so ask when unsure.

    ``by_me`` means "written by the subject" (you, in self mode)."""
    if getattr(args, "authorship", None):
        return args.authorship
    name = manifest.subject.name
    if not sys.stdin.isatty():
        if not manifest.is_self:  # never assume someone else's authorship
            raise ValueError(f"Say who wrote {what}: --authorship by_me ({name} wrote it, word for word) "
                             "or --authorship ai_assisted (anything else; never quoted)")
        print(style("2", f"Treating {what} as written by you (by_me). Use --authorship ai_assisted if not."))
        return "by_me"
    if manifest.is_self:
        print(f"Did you write {what} yourself, word for word?")
        print("  [y] yes, I wrote it        -> it can be quoted as your words")
        print("  [a] AI-assisted / drafted by someone else -> used as context only, never quoted as you")
        print("  [n] no, it isn't mine      -> don't import it")
    else:
        print(f"Did {name} write {what}, word for word?")
        print(f"  [y] yes, {name} wrote it   -> it can be quoted as {name}'s words")
        print(f"  [a] no: written about {name}, ghostwritten or AI-drafted -> context only, never quoted as {name}")
        print("  [n] not relevant          -> don't import it")
    while True:
        ans = input("> ").strip().lower()
        if ans in ("y", "yes") or (ans == "" and manifest.is_self):
            return "by_me"
        if ans in ("a", "ai"):
            return "ai_assisted"
        if ans in ("n", "no"):
            raise ValueError("Not imported.")


def _print_ingest(res, st) -> None:
    for s in res.added:
        tag = "" if s.authorship == "by_me" else style("33", " [ai_assisted]")
        print(f"  + {s.id}  {s.date or '          '}  {s.title}  ({s.chunks} chunks){tag}")
    for old in res.replaced:
        print(style("2", f"  ~ replaced older version {old}"))
    for p in res.skipped_duplicates:
        print(style("2", f"  = unchanged: {p}"))
    for p, err in res.errors:
        print(style("31", f"  ! {p}: {err}"), file=sys.stderr)
    print(f"{len(res.added)} added, {len(res.replaced)} updated, {len(res.skipped_duplicates)} unchanged, "
          f"{len(res.errors)} error(s). Corpus: {st['sources']} sources, {st['chunks']} chunks.")


def cmd_ingest(args) -> int:
    root = _root(args)
    manifest = load_manifest(root)
    manifest.require_consent()
    authorship = _ask_authorship(args, "these files", manifest)
    with Store(index_path(root)) as store:
        res = store.ingest_paths([Path(p) for p in args.paths], manifest.privacy.redact_names, args.date, authorship,
                                 redact_pii=manifest.privacy.redact_pii)
        _print_ingest(res, store.stats())
    return 1 if res.errors and not res.added else 0


def _looks_like_url(target: str) -> bool:
    return bool(re.match(r"^https?://", target, re.I)) or (
        not Path(target).exists() and bool(re.match(r"^[\w-]+(\.[\w-]+)+(/.*)?$", target)))


def cmd_add(args) -> int:
    target = args.target
    if not _looks_like_url(target):
        args.paths, args.date = [target], None
        return cmd_ingest(args)

    from .importers.site import ImportError_, canonical, import_site
    from .store import IngestResult

    root = _root(args)
    manifest = load_manifest(root)
    manifest.require_consent()
    whose = "your" if manifest.is_self else f"{manifest.subject.name}'s"
    print(f"Looking for {whose} writing on {target} (respecting robots.txt, about 1 request per second)...", flush=True)
    try:
        found = import_site(target, max_pages=args.max_pages, exclude=args.exclude, delay=args.delay,
                            progress=lambda n, total, u: print(style("2", f"  [{n}/{total}] {u}"), file=sys.stderr))
    except ImportError_ as exc:
        raise ValueError(str(exc)) from exc
    print(f"Found {len(found.pages)} page(s) via {', '.join(found.discovered_via)}; skipped {len(found.skipped)}.")
    if args.verbose or args.dry_run:
        for u, why in found.skipped:
            print(style("2", f"  - {u}  ({why})"))
    if args.dry_run:
        for pg in found.pages:
            print(f"  + {pg.date or '          '}  {pg.title}  ({len(pg.text.split())} words)  {pg.url}")
        print("Dry run: nothing was imported.")
        return 0
    if not found.pages:
        print("Nothing to import.")
        return 1
    authorship = _ask_authorship(args, f"the {len(found.pages)} page(s) on {found.site}", manifest)
    res = IngestResult([], [], [])
    with Store(index_path(root)) as store:
        for pg in found.pages:
            try:
                row = store.ingest_text(pg.url, pg.title, pg.date, pg.text, result=res,
                                        redact_names=manifest.privacy.redact_names,
                                        redact_pii=manifest.privacy.redact_pii, authorship=authorship,
                                        origin="site", ownership_proof="unverified_site", url=pg.url)
            except ValueError as exc:
                res.errors.append((pg.url, str(exc)))
                continue
            (res.added if row else res.skipped_duplicates).append(row or pg.url)
        _print_ingest(res, store.stats())
        gone = set(store.keys_with_prefix(canonical(found.site))) - {pg.url for pg in found.pages}
        gone -= {u for u, _ in found.skipped}
    if gone:
        print(style("33", f"{len(gone)} previously imported page(s) were not found this time "
                          "(deleted or moved?). Review with `anyone sources` and `anyone remove <id>`."))
    return 0


def cmd_authorship(args) -> int:
    root = _root(args)
    with Store(index_path(root)) as store:
        ok = store.set_authorship(args.source_id, args.value)
    print(f"{args.source_id} marked {args.value}" if ok else f"No source {args.source_id}")
    return 0 if ok else 1


def cmd_sources(args) -> int:
    root = _root(args)
    with Store(index_path(root)) as store:
        rows = store.list_sources()
    if not rows:
        print("No sources yet. Run: anyone ingest <path>")
        return 0
    for s in rows:
        flag = style("31", " [excluded]") if s.excluded else ""
        who = "" if s.authorship == "by_me" else style("33", f" [{s.authorship}]")
        print(f"{s.id}  {s.date or '----------'}  {s.title}  ({s.chunks} chunks){who}{flag}")
        print(style("2", f"    {s.path}  · {s.origin}, {s.ownership_proof.replace('_', ' ')}"))
    return 0


def _toggle(args, excluded: bool) -> int:
    root = _root(args)
    with Store(index_path(root)) as store:
        ok = store.set_excluded(args.source_id, excluded)
    print(("Excluded " if excluded else "Included ") + args.source_id if ok else f"No source {args.source_id}")
    return 0 if ok else 1


def cmd_remove(args) -> int:
    root = _root(args)
    with Store(index_path(root)) as store:
        ok = store.remove_source(args.source_id)
    print(f"Removed {args.source_id} from the index" if ok else f"No source {args.source_id}")
    return 0 if ok else 1


def cmd_ask(args) -> int:
    persona = Persona(_root(args), provider_override=args.provider)
    try:
        answer = persona.ask(args.question)
    finally:
        persona.close()
    print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False) if args.json else render(answer))
    return 0


def cmd_chat(args) -> int:
    persona = Persona(_root(args), provider_override=args.provider)
    convo = Conversation(persona)
    print(style("2", persona.manifest.disclosure))
    print(style("2", "Ask a question. Empty line or Ctrl-D to quit. Nothing is saved.\n"))
    try:
        while True:
            try:
                q = input(style("1", "you> ")).strip()
            except EOFError:
                print()
                break
            if not q:
                break
            try:
                print(render(convo.send(q)) + "\n")
            except ProviderError as exc:
                print(style("31", f"error: {exc}"), file=sys.stderr)
            except ValueError as exc:
                print(style("31", f"error: {exc}"), file=sys.stderr)
    finally:
        persona.close()
    return 0


def cmd_revoke(args) -> int:
    root = _root(args)
    if not args.yes:
        confirm = input("Delete the Anyone index (all ingested text, chunks and search data)? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return 1
    receipt = revoke(root)
    print(f"Revoked. Index deleted. Receipt: {receipt}\nYour original files were not touched.")
    return 0


def cmd_serve(args) -> int:
    from .web import serve

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(style("33", "warning: binding to a non-loopback address exposes this persona and your API budget "
                          "to your network. There is no authentication."), file=sys.stderr)
    serve(_root(args), host=args.host, port=args.port, provider_override=args.provider)
    return 0


def cmd_eval_init(args) -> int:
    from .evaluate import init_eval

    path = init_eval(_root(args))
    print(f"Created {path}. Write your own answers BEFORE looking at any AI output, then: anyone eval run")
    return 0


def cmd_eval_run(args) -> int:
    from .evaluate import ARMS, load_eval_set, planned_calls, run_eval, write_outputs
    from .providers import AnthropicProvider

    root = _root(args)
    items = load_eval_set(root)
    if args.limit:
        items = items[: args.limit]
    arms = args.arms.split(",") if args.arms else list(ARMS)
    persona = Persona(root, provider_override=args.provider)
    try:
        calls = planned_calls(items, arms, not args.no_judge)
        print(f"{len(items)} questions x {len(arms)} arms{' + judge' if not args.no_judge else ''}: "
              f"up to {calls} model calls with {persona.manifest.model.name}.", flush=True)
        if not args.yes and input("Run now? This uses your API credits. [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1
        llm = AnthropicProvider(persona.manifest.model)
        run = run_eval(persona, llm, items, arms, use_judge=not args.no_judge,
                       progress=lambda n, total, iid: print(f"  [{n}/{total}] {iid}", file=sys.stderr))
    finally:
        persona.close()
    out = write_outputs(root, run)
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"Saved to {out}\nFor the strongest signal, rate {out / 'rating_sheet.csv'} yourself (don't open rating_key.json),"
          f" then: anyone eval score {out}")
    return 0


def cmd_eval_score(args) -> int:
    from .evaluate import score_ratings

    result = score_ratings(Path(args.run_dir))
    if not result["arms"]:
        print("No ratings found. Fill in the rating_1_to_5 column first.")
        return 1
    print(f"Blind ratings across {result['items_compared']} questions:")
    for arm, r in result["arms"].items():
        print(f"  {arm:9} mean {r['mean_rating']}  win rate {r['win_rate']}  ({r['ratings']} ratings)")
    return 0


def cmd_mcp(args) -> int:
    try:
        from .mcp_server import run
    except ImportError:
        print("The MCP server needs the 'mcp' extra: pip install 'anyone-ai[mcp]'", file=sys.stderr)
        return 2
    run(_root(args), provider_override=args.provider)  # stdio: never print to stdout here
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anyone", description="Simulate anyone from public sources. "
                                "Every quote is verbatim, cited and machine-checked.")
    p.add_argument("--version", action="version", version=f"anyone {__version__}")
    p.add_argument("-C", "--dir", default=".", help="persona directory (default: current; searches parents)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="create persona.yaml for the person to simulate")
    s.add_argument("--name", required=True, help="the person's name as it should appear")
    s.add_argument("--kind", choices=SUBJECT_KINDS, default="self",
                   help="who this is: public_figure, private_individual or self (default: self)")
    s.add_argument("--i-am-this-person", action="store_true", help="--kind self: confirm self-attestation non-interactively")
    s.add_argument("--yes", action="store_true", help="other kinds: accept the notice non-interactively")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("ingest", help="add writing (.md .txt .html .pdf .docx files or folders)")
    s.add_argument("paths", nargs="+")
    s.add_argument("--date", help="date for these sources (YYYY-MM-DD) if not in front matter")
    s.add_argument("--authorship", choices=["by_me", "ai_assisted"],
                   help="by_me: written by the subject (you, in self mode); asked if omitted")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("add", help="add content: a website/blog URL, a file or a folder")
    s.add_argument("target", help="e.g. theirname.com, https://name.substack.com, ~/writing")
    s.add_argument("--authorship", choices=["by_me", "ai_assisted"],
                   help="by_me: written by the subject (you, in self mode); asked if omitted")
    s.add_argument("--max-pages", type=int, default=200)
    s.add_argument("--exclude", action="append", default=[], metavar="REGEX", help="skip URLs matching this (repeatable)")
    s.add_argument("--dry-run", action="store_true", help="show what would be imported, import nothing")
    s.add_argument("--delay", type=float, default=1.0, help="seconds between requests (robots.txt crawl-delay wins if larger)")
    s.add_argument("-v", "--verbose", action="store_true", help="list skipped pages and why")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("authorship", help="mark a source as written by the subject (by_me) or not (ai_assisted)")
    s.add_argument("source_id")
    s.add_argument("value", choices=["by_me", "ai_assisted"])
    s.set_defaults(func=cmd_authorship)

    sub.add_parser("sources", help="list ingested sources").set_defaults(func=cmd_sources)
    for name, fn, helptext in (("exclude", lambda a: _toggle(a, True), "stop using a source in answers"),
                               ("include", lambda a: _toggle(a, False), "use an excluded source again"),
                               ("remove", cmd_remove, "delete a source from the index")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("source_id")
        s.set_defaults(func=fn)

    for name, fn, helptext in (("ask", cmd_ask, "ask one question"), ("chat", cmd_chat, "multi-turn conversation")):
        s = sub.add_parser(name, help=helptext)
        if name == "ask":
            s.add_argument("question")
            s.add_argument("--json", action="store_true", help="machine-readable output")
        s.add_argument("--provider", choices=["anthropic", "extractive"], help="override model.provider")
        s.set_defaults(func=fn)

    s = sub.add_parser("serve", help="local web chat with an evidence panel")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--provider", choices=["anthropic", "extractive"], help="override model.provider")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("mcp", help="serve this persona to AI agents over MCP (stdio)")
    s.add_argument("--provider", choices=["anthropic", "extractive"], help="override model.provider")
    s.set_defaults(func=cmd_mcp)

    ev = sub.add_parser("eval", help="measure how well the simulation matches your own answers")
    evsub = ev.add_subparsers(dest="eval_command", required=True)
    evsub.add_parser("init", help="create eval.yaml for your questions and answers").set_defaults(func=cmd_eval_init)
    s = evsub.add_parser("run", help="run all arms, judge them, write a report and a blind rating sheet")
    s.add_argument("--arms", help="comma-separated subset of: anyone,bio_only,generic")
    s.add_argument("--no-judge", action="store_true", help="skip LLM judging (blind sheet only)")
    s.add_argument("--limit", type=int, help="only the first N questions")
    s.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    s.add_argument("--provider", choices=["anthropic", "extractive"], help="override provider for the anyone arm")
    s.set_defaults(func=cmd_eval_run)
    s = evsub.add_parser("score", help="score your filled-in blind rating sheet")
    s.add_argument("run_dir")
    s.set_defaults(func=cmd_eval_score)

    s = sub.add_parser("revoke", help="delete the index and write a revocation receipt")
    s.add_argument("--yes", action="store_true", help="skip confirmation")
    s.set_defaults(func=cmd_revoke)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ManifestError, ProviderError, ValueError) as exc:
        print(style("31", f"error: {exc}"), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
