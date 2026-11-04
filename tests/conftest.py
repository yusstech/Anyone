import shutil
from pathlib import Path

import pytest

from citeself.manifest import DATA_DIR
from citeself.providers import GenerationRequest, GenerationResult
from citeself.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


class FakeProvider:
    """Scripted provider. ``script`` is a callable(request) -> GenerationResult or a static result."""
    name = "fake"

    def __init__(self, script):
        self.script = script
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return self.script(request) if callable(self.script) else self.script


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    root = tmp_path / "avery"
    shutil.copytree(FIXTURES / "avery", root)
    with Store(root / DATA_DIR / "index.db") as store:
        res = store.ingest_paths([root / "sources"], redact_names=["Jordan Lee"])
        assert len(res.added) == 4 and not res.errors
    return root
