import pytest
from src.prompts.one_shot_prompt import OneShotPromptHandler

analyzer = OneShotPromptHandler()

def test_simple_summary():
    text = "The iPhone 15 was launched yesterday. It features a better camera and faster processor."
    result = analyzer.summarize_text(text)
    assert "summary" in result
    assert result["confidence"] <= 1

def test_long_text_summary():
    text = "Climate change is one of the biggest global challenges. Rising temperatures and sea levels impact agriculture, wildlife, and human settlements."
    result = analyzer.summarize_text(text)
    assert "climate" in result["summary"].lower()

def test_short_text_summary():
    text = "Elon Musk founded SpaceX."
    result = analyzer.summarize_text(text)
    assert "spacex" in result["summary"].lower()
