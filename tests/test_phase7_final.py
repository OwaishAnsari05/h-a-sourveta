from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]

def read(path): return (ROOT/path).read_text(encoding="utf-8").lower()

def test_multilingual_indexing_stack_is_aligned():
    processing=read("api/services/processing.py"); chroma=read("vectorstore/chroma_store.py")
    assert "sourveta_multilingual" in processing and "intfloat/multilingual-e5-base" in processing and "encode_passages" in processing
    assert "sourveta_multilingual" in chroma and "intfloat/multilingual-e5-base" in chroma

def test_api_has_cors_and_root():
    main=read("api/main.py")
    assert "corsmiddleware" in main
    assert "allow_origins=get_allowed_origins()" in main
    assert "def root()" in main

def test_docker_build_does_not_require_local_data():
    docker=read("dockerfile")
    assert "copy data ./data" not in docker
    assert "mkdir -p data/documents data/chroma_db data/chunks" in docker

def test_production_indexing_is_not_tata_specific():
    assert 'collection_name="tata_annual_report"' not in read("api/services/processing.py")
    assert 'collection_name="tata_annual_report"' not in read("vectorstore/chroma_store.py")

def test_phase7_files_parse():
    for rel in ["api/main.py","api/routes/health.py","api/services/processing.py","vectorstore/chroma_store.py"]:
        ast.parse(read(rel))
