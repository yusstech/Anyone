"""citeself command-line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .engine import Answer, Conversation, Persona
from .governance import revoke
from .manifest import DATA_DIR, ManifestError, find_root, load_manifest, write_new_manifest
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


def cmd_init(args) -> int:
    root = Path(args.dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.i_am_this_person:
        print("CiteSelf only builds a simulation of YOURSELF, from material you wrote or have the right to use.")
        confirm = input(f"Are you {args.name}, and do you agree to that? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted. Nothing was created.")
            return 1
    path = write_new_manifest(root, args.name)
    print(f"Created {path}\nNext: citeself ingest <your-writing-folder>")
    return 0


def cmd_ingest(args) -> int:
    root = _root(args)
    manifest = load_manifest(root)
    manifest.require_consent()
    with Store(root / DATA_DIR / "index.db") as store:
        res = store.ingest_paths([Path(p) for p in args.paths], manifest.privacy.redact_names, args.date)
        for s in res.added:
            print(f"  + {s.id}  {s.title}  ({s.chunks} chunks)")
        for old in res.replaced:
            print(style("2", f"  ~ replaced older version {old}"))
        for p in res.skipped_duplicates:
            print(style("2", f"  = already ingested: {p}"))
        for p, err in res.errors:
            print(style("31", f"  ! {p}: {err}"), file=sys.stderr)
        st = store.stats()
    print(f"{len(res.added)} added, {len(res.skipped_duplicates)} duplicate(s), {len(res.errors)} error(s). "
          f"Corpus: {st['sources']} sources, {st['chunks']} chunks.")
    return 1 if res.errors and not res.added else 0


def cmd_sources(args) -> int:
    root = _root(args)
    with Store(root / DATA_DIR / "index.db") as store:
        rows = store.list_sources()
    if not rows:
        print("No sources yet. Run: citeself ingest <path>")
        return 0
    for s in rows:
        flag = style("31", " [excluded]") if s.excluded else ""
        print(f"{s.id}  {s.date or '----------'}  {s.title}  ({s.chunks} chunks){flag}\n    {style('2', s.path)}")
    return 0


def _toggle(args, excluded: bool) -> int:
    root = _root(args)
    with Store(root / DATA_DIR / "index.db") as store:
        ok = store.set_excluded(args.source_id, excluded)
    print(("Excluded " if excluded else "Included ") + args.source_id if ok else f"No source {args.source_id}")
    return 0 if ok else 1


def cmd_remove(args) -> int:
    root = _root(args)
    with Store(root / DATA_DIR / "index.db") as store:
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
        confirm = input("Delete the CiteSelf index (all ingested text, chunks and search data)? [y/N] ").strip().lower()
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


def cmd_mcp(args) -> int:
    try:
        from .mcp_server import run
    except ImportError:
        print("The MCP server needs the 'mcp' extra: pip install 'citeself[mcp]'", file=sys.stderr)
        return 2
    run(_root(args), provider_override=args.provider)  # stdio: never print to stdout here
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeself", description="An AI of you that only speaks on the record.")
    p.add_argument("--version", action="version", version=f"citeself {__version__}")
    p.add_argument("-C", "--dir", default=".", help="persona directory (default: current; searches parents)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="create persona.yaml for yourself")
    s.add_argument("--name", required=True, help="your name as it should appear")
    s.add_argument("--i-am-this-person", action="store_true", help="confirm self-attestation non-interactively")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("ingest", help="add your writing (.md .txt .html .pdf files or folders)")
    s.add_argument("paths", nargs="+")
    s.add_argument("--date", help="date for these sources (YYYY-MM-DD) if not in front matter")
    s.set_defaults(func=cmd_ingest)

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
