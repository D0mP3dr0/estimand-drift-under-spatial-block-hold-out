"""Check that the published JSON records carry the same numbers as the originals.

The records were published with their free-text notes translated into English, the names
of internal note files and of the authors' notes and scratch directories replaced by
neutral ones (internal/...), and the values of a few attribution keys (who produced or
requested a record, and when) replaced by neutral labels. Everything else is untouched.
This script computes, for each JSON or JSON Lines file listed in PAYLOAD_SHA256, the
sha256 of its "payload": the canonical JSON with every free-text string replaced by a
fixed placeholder. A string counts as free text when it has four or more words, names a
.md note, is one of the neutral internal/... names, or is the value of one of the
attribution keys listed in ROLE_KEYS. Numbers, booleans, nulls, keys, structure, short
labels, identifiers, data paths and sha256 values all enter the digest.

PAYLOAD_SHA256 lists the digest of each ORIGINAL record, computed before publication.
A match here shows that the cleaning changed prose and attribution labels only.

Usage (from the repository root):  python verify_payload.py
"""
import hashlib
import json
import pathlib
import sys

PLACEHOLDER = "<text>"


ROLE_KEYS = {"autor", "produtor", "frente", "convocado_por", "fio", "quando", "falta", "entrega"}


def is_free_text(s, key=None):
    return len(s.split()) >= 4 or ".md" in s or key in ROLE_KEYS or "internal/" in s


def mask(o, key=None):
    if isinstance(o, dict):
        return {k: mask(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [mask(v, key) for v in o]
    if isinstance(o, str) and is_free_text(o, key):
        return PLACEHOLDER
    return o


def payload_sha256(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".jsonl"):
        docs = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        docs = [json.loads(text)]
    canon = "\n".join(json.dumps(mask(d), sort_keys=True, ensure_ascii=False, separators=(",", ":")) for d in docs)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def main():
    root = pathlib.Path(__file__).resolve().parent
    expected = {}
    for line in (root / "PAYLOAD_SHA256").read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            expected[rel.strip()] = digest
    bad = 0
    for rel, digest in sorted(expected.items()):
        ok = payload_sha256(root / rel) == digest
        bad += not ok
        if not ok:
            print(f"MISMATCH {rel}")
    print(f"{len(expected) - bad}/{len(expected)} records: payload identical to the original")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
