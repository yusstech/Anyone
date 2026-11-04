"""Revocation: delete every derived artifact and leave a receipt."""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

from .manifest import DATA_DIR, load_manifest
from .store import Store


def revoke(root: Path) -> Path:
    """Delete the index (sources, chunks, search index) under .citeself/ and write a receipt.

    The owner's original files are never touched: they belong to the owner, and
    CiteSelf only ever read them. Conversations are not stored, so there is
    nothing else to delete. Returns the receipt path.
    """
    manifest = load_manifest(root)
    data_dir = root / DATA_DIR
    db = data_dir / "index.db"
    stats, hashes = {"sources": 0, "excluded": 0, "chunks": 0}, []
    if db.exists():
        with Store(db) as store:
            stats, hashes = store.stats(), store.source_hashes()
    if data_dir.exists():
        shutil.rmtree(data_dir)
    now = _dt.datetime.now(_dt.timezone.utc)
    receipt = {
        "type": "citeself.revocation",
        "persona": manifest.persona.name,
        "revoked_at": now.isoformat(timespec="seconds"),
        "deleted": {"directory": str(data_dir), **stats},
        # Hashes let the owner confirm *which* material was removed without the receipt containing it.
        "deleted_source_sha256": hashes,
        "not_deleted": "Original source files supplied by the owner were not modified.",
        "conversation_logs": "none stored",
    }
    path = root / f"revocation-receipt-{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path
