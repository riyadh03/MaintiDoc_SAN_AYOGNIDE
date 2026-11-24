"""Tests bout-en-bout du contrat de citation : packer → prompt → validation."""
from maintidoc.generation.ollama_client import format_evidence, validate_citations
from maintidoc.retrieval.evidence_packer import EvidencePacker


def _pack(make_hit):
    hits = [make_hit("p03_text_01", 1, page=3, content="LED orange fixe = défaut PSU")]
    return EvidencePacker().pack(hits, [])


def test_prompt_labels_match_allowed_citations(make_hit):
    """Régression du bug initial : le prompt doit exposer exactement les ids autorisés."""
    texts, images, citations = _pack(make_hit)

    prompt = format_evidence(texts, images)

    for cid in citations:
        assert f"[SOURCE {cid} |" in prompt
    assert "LED orange fixe" in prompt


def test_valid_citation_passes(make_hit):
    _, _, citations = _pack(make_hit)
    output = {"status": "answered", "citations": ["SAN-MAN-001:p3:p03_text_01"]}
    assert validate_citations(output, citations) == []


def test_citation_with_brackets_is_normalized(make_hit):
    _, _, citations = _pack(make_hit)
    output = {"status": "answered", "citations": ["[SOURCE SAN-MAN-001:p3:p03_text_01]"]}
    assert validate_citations(output, citations) == []


def test_hallucinated_citation_is_flagged(make_hit):
    _, _, citations = _pack(make_hit)
    output = {"status": "answered", "citations": ["SAN-MAN-001:p7:p07_text_01"]}
    issues = validate_citations(output, citations)
    assert len(issues) == 1 and "p07_text_01" in issues[0]


def test_answer_without_citation_is_flagged(make_hit):
    _, _, citations = _pack(make_hit)
    assert validate_citations({"status": "answered", "citations": []}, citations)


def test_insufficient_evidence_without_citation_is_ok(make_hit):
    _, _, citations = _pack(make_hit)
    assert validate_citations({"status": "insufficient_evidence", "citations": []}, citations) == []


def test_malformed_outputs():
    assert validate_citations("pas un dict", {})
    assert validate_citations({"citations": "SAN-MAN-001:p3:x"}, {})
