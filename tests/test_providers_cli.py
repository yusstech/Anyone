import json
import os
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from anyone_ai.cli import main
from anyone_ai.manifest import DATA_DIR, ModelSettings
from anyone_ai.providers import FALLBACK_BETA, AnthropicProvider, GenerationRequest, MalformedOutput, ProviderError

from .conftest import FIXTURES


class FakeMessages:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.response


def fake_client(response=None, exc=None):
    msgs = FakeMessages(response, exc)
    return SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs)), msgs


def response(text=None, stop_reason="end_turn"):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(content=content, stop_reason=stop_reason, model="claude-opus-5-5", stop_details=None,
                           usage=SimpleNamespace(input_tokens=5, output_tokens=7, cache_read_input_tokens=0,
                                                 cache_creation_input_tokens=0))


REQ = GenerationRequest(system="sys", messages=[{"role": "user", "content": "q"}], evidence={})


def test_anthropic_request_shape():
    client, msgs = fake_client(response(json.dumps({"segments": [{"kind": "no_evidence", "text": "x", "evidence_ids": []}]})))
    result = AnthropicProvider(ModelSettings(), client=client).generate(REQ)
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["output_config"]["format"]["type"] == "json_schema" and call["output_config"]["effort"] == "medium"
    assert "cache_control" not in call and "temperature" not in call and "thinking" not in call
    assert result.segments[0]["kind"] == "no_evidence" and result.usage["output_tokens"] == 7


def test_anthropic_without_fallback_uses_plain_endpoint():
    client, msgs = fake_client(response('{"segments": []}'))
    AnthropicProvider(ModelSettings(refusal_fallback=False), client=client).generate(REQ)
    assert "betas" not in msgs.calls[0] and "fallbacks" not in msgs.calls[0]


def test_anthropic_refusal_max_tokens_and_bad_json():
    client, _ = fake_client(response(stop_reason="refusal"))
    assert AnthropicProvider(ModelSettings(), client=client).generate(REQ).refused
    for resp in (response("{}", stop_reason="max_tokens"), response("not json"), response(None)):
        client, _ = fake_client(resp)
        with pytest.raises(MalformedOutput):
            AnthropicProvider(ModelSettings(), client=client).generate(REQ)


def test_anthropic_errors_become_safe_messages():
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    auth = anthropic.AuthenticationError("bad key sk-secret", response=httpx2.Response(401, request=req), body=None)
    conn = anthropic.APIConnectionError(request=req)
    for exc, expected in ((auth, "credentials"), (conn, "network")):
        client, _ = fake_client(exc=exc)
        with pytest.raises(ProviderError) as info:
            AnthropicProvider(ModelSettings(), client=client).generate(REQ)
        assert expected in str(info.value) and "sk-secret" not in str(info.value)


def test_cli_end_to_end_offline(tmp_path, capsys):
    root = tmp_path / "me"
    assert main(["-C", str(root), "init", "--name", "Test Person", "--i-am-this-person"]) == 0
    assert (root / "persona.yaml").exists()
    assert main(["-C", str(root), "ingest", str(FIXTURES / "avery" / "sources")]) == 0
    assert main(["-C", str(root), "sources"]) == 0
    capsys.readouterr()
    assert main(["-C", str(root), "ask", "release quality and rollback", "--provider", "extractive", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ai_generated"] and out["segments"][0]["kind"] == "quote" and out["segments"][0]["verified"]

    assert main(["-C", str(root), "revoke", "--yes"]) == 0
    assert not (root / DATA_DIR).exists()
    receipt = json.loads(next(root.glob("revocation-receipt-*.json")).read_text())
    assert receipt["deleted"]["sources"] == 4 and len(receipt["deleted_source_sha256"]) == 4
    assert (FIXTURES / "avery" / "sources" / "2026-03-shipping-talk.md").exists()


def test_cli_errors_are_reported_not_raised(tmp_path, capsys):
    assert main(["-C", str(tmp_path), "ask", "hello"]) == 2
    assert "persona.yaml" in capsys.readouterr().err


@pytest.mark.live
def test_live_anthropic_smoke(persona_dir):
    """Real API call. Run with: pytest -m live (needs ANTHROPIC_API_KEY)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from anyone_ai.engine import Persona

    p = Persona(persona_dir)
    a = p.ask("How does Avery think about releasing with known risks?")
    assert a.status in ("ok", "no_evidence"), a.to_dict()
    assert all(s.verified for s in a.segments if s.kind == "quote")
    print(json.dumps(a.to_dict(), indent=2))
