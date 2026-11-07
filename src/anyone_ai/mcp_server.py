"""MCP server: let any MCP-capable agent consult an Anyone persona with citations.

Run with ``anyone mcp`` (stdio). Requires the ``mcp`` extra.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .engine import Persona
from .providers import Provider, ProviderError


def build_server(root: Path, provider: Provider | None = None, provider_override: str | None = None) -> MCPServer:
    persona = Persona(root, provider=provider, provider_override=provider_override)
    name = persona.manifest.subject.name
    lock = threading.Lock()  # one SQLite connection; serialize access from worker threads

    server = MCPServer(
        name="anyone",
        log_level="WARNING",
        title=f"Anyone: {name}",
        instructions=(
            f"{persona.manifest.disclosure} When you relay answers, keep the labels: present 'quote' segments as "
            f"{name}'s verbatim words with their source, 'extrapolation' segments as inference (not {name}'s words), "
            "and 'no_evidence' segments as gaps. Never upgrade an extrapolation into a quote."
        ),
    )

    def _ask(question: str) -> dict:
        with lock:
            try:
                return persona.ask(question).to_dict()
            except ProviderError as exc:
                return {"error": str(exc), "disclosure": persona.manifest.disclosure, "ai_generated": True}
            except ValueError as exc:
                return {"error": str(exc)}

    @server.tool(
        title=f"Ask {name}'s AI simulation",
        description=(
            f"Ask an AI simulation of {name}, built from {persona.manifest.source_basis}. Returns labeled segments: "
            "'quote' (verbatim, machine-verified against the cited source), 'extrapolation' (inference, cites the "
            "evidence it relies on) and 'no_evidence'. Each call is independent (no memory between calls)."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
    )
    async def ask_persona(question: str) -> dict:
        return await asyncio.to_thread(_ask, question)

    @server.tool(
        title=f"Search {name}'s record",
        description=(f"Full-text search over {persona.manifest.source_basis}. Returns verbatim passages with source "
                     "titles and dates. No AI generation; use this to check exactly what was said."),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def search_record(query: str, limit: int = 5) -> dict:
        limit = max(1, min(int(limit), 20))

        def run() -> dict:
            with lock:
                hits = persona.store.search(query, k=limit)
            return {"persona": name, "passages": [
                {"source_id": h.source_id, "title": h.title, "date": h.date, "text": h.text} for h in hits]}

        return await asyncio.to_thread(run)

    @server.tool(
        title=f"About {name}'s simulation",
        description="Who this simulation represents, the disclosure to show users, its sources and its boundaries.",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def persona_info() -> dict:
        def run() -> dict:
            with lock:
                sources = [s for s in persona.store.list_sources() if not s.excluded]
            m = persona.manifest
            return {
                "name": name, "subject_kind": m.subject.kind, "bio": m.subject.bio, "disclosure": m.disclosure,
                "blocked_topics": m.boundaries.blocked_topics,
                "extrapolation_allowed": m.boundaries.allow_extrapolation,
                "sources": [{"source_id": s.id, "title": s.title, "date": s.date} for s in sources],
            }

        return await asyncio.to_thread(run)

    return server


def run(root: Path, provider_override: str | None = None) -> None:
    build_server(root, provider_override=provider_override).run("stdio")
