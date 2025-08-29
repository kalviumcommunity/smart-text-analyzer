import pytest
from src.prompts.structured_output import StructuredPromptHandler

analyzer = StructuredPromptHandler()

def test_positive_text():
    text = "I absolutely love this AI project!"
    result = analyzer.analyze_text(text)
    assert result["sentiment"] == "positive"
    assert 0 <= result["confidence"] <= 1

def test_negative_text():
    text = "This tool is useless and frustrating."
    result = analyzer.analyze_text(text)
    assert result["sentiment"] == "negative"

def test_summary_and_category():
    text = "Cristiano Ronaldo scored a goal for Portugal in the World Cup."
    result = analyzer.analyze_text(text)
    assert "summary" in result
    assert result["category"].lower() in ["sports", "football", "news"]
