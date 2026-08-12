"""
Stage 15 — Automated tests: NLP pipeline.

Run with: pytest tests/test_nlp.py -v
"""
import pytest
from nlp.preprocessing import clean_text, tokenize_lemmatize
from nlp.duplicate_detection import DuplicateIndex, pairwise_duplicate_groups
from nlp.ner import extract_assistance_types
from nlp.synthetic_data import generate, CATEGORIES


class TestPreprocessing:
    def test_clean_text_lowercases(self):
        assert clean_text("HELLO World") == "hello world"

    def test_clean_text_strips_urls(self):
        result = clean_text("Check http://example.com/path now")
        assert "http" not in result and "example.com" not in result

    def test_clean_text_strips_mentions(self):
        result = clean_text("Help needed @rescueteam now")
        assert "@rescueteam" not in result

    def test_clean_text_handles_empty(self):
        assert clean_text("") == ""

    def test_clean_text_handles_non_string(self):
        assert clean_text(None) == ""


class TestDuplicateDetection:
    def test_identical_messages_flagged(self):
        idx = DuplicateIndex()
        idx.add("MSG-1", "Our street is flooded near Koramangala, need rescue")
        result = idx.check_duplicate("Our street is flooded near Koramangala, need rescue")
        assert result["is_duplicate"] is True
        assert result["similarity"] > 0.95

    def test_unrelated_messages_not_flagged(self):
        idx = DuplicateIndex()
        idx.add("MSG-1", "Fire broke out near Whitefield, people trapped")
        result = idx.check_duplicate("Requesting food packets for the shelter camp")
        assert result["is_duplicate"] is False

    def test_empty_index_returns_no_duplicate(self):
        idx = DuplicateIndex()
        result = idx.check_duplicate("Any message at all")
        assert result["is_duplicate"] is False
        assert result["best_match_id"] is None

    def test_pairwise_groups_finds_clusters(self):
        ids = ["A", "B", "C"]
        texts = [
            "Our street is flooded near Koramangala, need rescue",
            "Our street is flooded near Koramangala need rescue",  # near-dup of A
            "Requesting food packets for the shelter camp",
        ]
        groups = pairwise_duplicate_groups(ids, texts, threshold=0.6)
        assert any(set(g) == {"A", "B"} for g in groups)


class TestNER:
    def test_medical_keywords_detected(self):
        types = extract_assistance_types("Need an ambulance urgently, person is bleeding")
        assert "medical" in types

    def test_multiple_assistance_types(self):
        types = extract_assistance_types("Need food and clean drinking water urgently")
        assert "food" in types and "water" in types

    def test_no_keywords_returns_empty(self):
        types = extract_assistance_types("Just saying hello to everyone")
        assert types == []


class TestSyntheticData:
    def test_generate_produces_requested_count(self):
        records = generate(n_messages=100)
        assert len(records) == 100

    def test_generate_covers_all_categories_eventually(self):
        records = generate(n_messages=500)
        seen_categories = {r["category"] for r in records}
        assert seen_categories == set(CATEGORIES)

    def test_generate_tags_source_synthetic(self):
        records = generate(n_messages=20)
        assert all(r["source"] == "synthetic" for r in records)
