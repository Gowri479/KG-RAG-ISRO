from src.preprocessing.clean import clean_markdown
from src.preprocessing.chunk import chunk_text


def test_clean_markdown_removes_noise_and_deduplicates():
    raw_markdown = """
    # ISRO Mission Portal
    
    **Mission:** Chandrayaan-3
    
    ---
    
    Chandrayaan-3
    
    [1] ISRO official website
    
    Chandrayaan-3
    """

    cleaned = clean_markdown(raw_markdown, source_url="https://example.com/isro")

    assert "Mission" in cleaned
    assert "Chandrayaan-3" in cleaned
    assert "---" not in cleaned
    assert cleaned.count("Chandrayaan-3") == 1
    assert cleaned.startswith("Mission") or "Chandrayaan-3" in cleaned


def test_chunk_text_creates_overlapping_chunks():
    text = " ".join(f"token{i}" for i in range(200))

    chunks = chunk_text(text, chunk_size=50, stride=10)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 50 for chunk in chunks)
    assert chunks[0] != chunks[1]
