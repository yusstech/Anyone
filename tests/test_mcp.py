import json

import pytest

pytest.importorskip("mcp")
from mcp.client import Client  # noqa: E402

from anyone_ai.mcp_server import build_server  # noqa: E402
from anyone_ai.providers import ExtractiveProvider  # noqa: E402


def payload(result):
    if getattr(result, "structured_content", None):
        return result.structured_content
    return json.loads(result.content[0].text)


@pytest.mark.anyio
async def test_mcp_tools(persona_dir):
    server = build_server(persona_dir, provider=ExtractiveProvider())
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"ask_persona", "search_record", "persona_info"}

        info = payload(await client.call_tool("persona_info", {}))
        assert info["name"] == "Avery Moreno" and len(info["sources"]) == 4 and "not Avery" in info["disclosure"]

        ans = payload(await client.call_tool("ask_persona", {"question": "release quality rollback"}))
        assert ans["ai_generated"] and ans["segments"][0]["kind"] == "quote" and ans["segments"][0]["verified"]

        blocked = payload(await client.call_tool("ask_persona", {"question": "what is your password"}))
        assert blocked["status"] == "boundary"

        hits = payload(await client.call_tool("search_record", {"query": "decision memo", "limit": 99}))
        assert hits["passages"] and "decision memo" in hits["passages"][0]["text"]
