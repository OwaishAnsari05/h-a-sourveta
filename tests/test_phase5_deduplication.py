from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_hashing_is_deterministic():
    from api.services.hashing import sha256_bytes
    assert sha256_bytes(b"SOURVETA")==sha256_bytes(b"SOURVETA")
    assert sha256_bytes(b"SOURVETA")!=sha256_bytes(b"sourveta")

def test_status_stores_hash_and_size(tmp_path,monkeypatch):
    from api.services import status
    path=tmp_path/"status.json"
    monkeypatch.setattr(status,"STATUS_PATH",str(path))
    item=status.create_status("doc1","a.pdf",sha256="abc",size_bytes=123,storage_filename="doc1_a.pdf")
    assert item["sha256"]=="abc" and item["size_bytes"]==123
    assert status.find_by_sha256("abc")["document_id"]=="doc1"

def test_phase5_upload_route_has_hash_dedup():
    text=(ROOT/"api/routes/documents.py").read_text(encoding="utf-8")
    assert "sha256_bytes" in text and "find_document_by_hash" in text and 'duplicate":True' in text

def test_phase5_lifecycle_endpoints_exist():
    text=(ROOT/"api/routes/documents.py").read_text(encoding="utf-8")
    assert '"/{document_id}/retry"' in text and 'router.delete("/{document_id}")' in text

def test_phase5_schema_exposes_document_identity():
    text=(ROOT/"api/schemas.py").read_text(encoding="utf-8")
    assert "sha256:str|None=None" in text and "duplicate:bool=False" in text