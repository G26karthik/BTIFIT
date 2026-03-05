"""Tests for BotTrainer.modules.entity_extractor."""

from BotTrainer.modules.entity_extractor import EntityExtractor


def test_extract_returns_entities(mock_llm):
    def fake_query(*args, **kwargs):
        return '{"city": "Mumbai", "departure": "Delhi"}'

    mock_llm.query = fake_query
    extractor = EntityExtractor(mock_llm)
    result = extractor.extract("Book a flight to Mumbai from Delhi", "book_flight")
    assert result == {"city": "Mumbai", "departure": "Delhi"}


def test_extract_caches_result(mock_llm):
    call_count = 0

    def fake_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return '{"city": "Mumbai"}'

    mock_llm.query = fake_query
    extractor = EntityExtractor(mock_llm)

    extractor.extract("Book flight to Mumbai", "book_flight")
    extractor.extract("Book flight to Mumbai", "book_flight")  # should be cached
    assert call_count == 1


def test_extract_returns_empty_on_bad_json(mock_llm):
    def fake_query(*args, **kwargs):
        return "not json {{{}"

    mock_llm.query = fake_query
    extractor = EntityExtractor(mock_llm)
    result = extractor.extract("something", "weather")
    assert result == {}
