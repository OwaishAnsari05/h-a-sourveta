import pytest
from api.config import MAX_UPLOAD_BYTES
from api.services.security import sanitize_filename,validate_pdf_bytes
from api.services.rate_limit import allow_request

def test_filename_is_path_safe():
    name=sanitize_filename(r"..\secret/../../annual report?.pdf")
    assert "/" not in name and "\\" not in name and name.endswith(".pdf")

def test_pdf_signature_is_required():
    validate_pdf_bytes(b"%PDF-1.7\nvalid")
    with pytest.raises(ValueError): validate_pdf_bytes(b"not a pdf")

def test_upload_limit_is_enforced():
    with pytest.raises(ValueError): validate_pdf_bytes(b"%PDF-1.7"+b"x"*MAX_UPLOAD_BYTES)

def test_rate_limit_is_bounded():
    key="phase6-test-client"
    assert allow_request(key,limit=2) is True
    assert allow_request(key,limit=2) is True
    assert allow_request(key,limit=2) is False