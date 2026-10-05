from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    return (ROOT/path).read_text(encoding="utf-8").lower()

def test_api_branding_is_generic():
    assert "tata annual report" not in read("api/main.py")
    assert "tata-annual-report" not in read("api/routes/health.py")

def test_ingestion_has_no_tata_defaults():
    assert "tata_annual_report" not in read("ingestion/pdf_parser.py")
    assert "tata_annual_report" not in read("ingestion/chunker.py")
    assert "tata industries limited" not in read("ingestion/pdf_parser.py")

def test_documents_service_has_no_tata_special_case():
    assert 'document_id == "tata_annual_report_2024_25"' not in read("api/services/documents.py")

def test_reranker_has_no_tata_specific_boost():
    assert "tata sons private limited" not in read("vectorstore/reranker.py")
    assert "tata chemicals limited" not in read("vectorstore/reranker.py")

def test_core_citation_logic_is_generic():
    text=read("generation/citation.py")
    assert "tata sons private limited" not in text
    assert "tata chemicals limited" not in text

def test_rag_has_no_benchmark_specific_logic():
    text=read("generation/rag.py")
    for term in ("tata", "microbiology", "bacteria", "structure_of_bacteria", "capsulk", "spuct of bacte8ia"):
        assert term not in text
