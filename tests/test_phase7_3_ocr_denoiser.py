from unittest.mock import patch
from ingestion.ocr_denoiser import corruption_ratio,should_denoise,denoise_page,_valid_candidate

def test_corruption_ratio_detects_embedded_digits():
    assert corruption_ratio("Bacte8ia Mes0tomAs")>0

def test_clean_text_does_not_trigger_denoising():
    needs,ratio,disagreement=should_denoise("This is clean document text with normal words.",ocr_confidence=92,ocr_used=True)
    assert needs is False
    assert ratio==0.0
    assert disagreement==0.0

def test_low_ocr_confidence_triggers_generic_denoising():
    needs,_,_=should_denoise("Spuct of Bacte8ia and Cull wa with additional extracted text to cross the minimum length gate.",ocr_confidence=41,ocr_used=True)
    assert needs is True

def test_candidate_safety_rejects_number_changes():
    assert _valid_candidate("Amount 123.45 was reported.","Amount 124.45 was reported.") is False

def test_denoiser_uses_same_page_alternative_and_returns_cleaned_text():
    from types import SimpleNamespace
    response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Structure of Bacteria\nCell wall\nFlagella"))])
    class Completions:
        @staticmethod
        def create(**kwargs):
            assert "ALTERNATIVE OCR EXTRACTION FROM THE SAME PAGE" in kwargs["messages"][1]["content"]
            return response
    client=SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    with patch("ingestion.ocr_denoiser._client",return_value=client):
        result=denoise_page("Spuct of Bacte8ia\nCull wa\nHagella\nThis is additional page text for the denoiser gate.",ocr_text="Structure of Bacteria\nCell wall\nFlagella\nThis is additional page text for the denoiser gate.",ocr_confidence=40,ocr_used=True)
    assert result["denoised"] is True
    assert "Structure of Bacteria" in result["text"]

def test_denoiser_falls_back_on_api_failure():
    with patch("ingestion.ocr_denoiser._client",side_effect=RuntimeError("offline")):
        result=denoise_page("Spuct of Bacte8ia with additional extracted page text for fallback testing.",ocr_confidence=40,ocr_used=True)
    assert result["denoised"] is False
    assert result["text"].startswith("Spuct of Bacte8ia")
