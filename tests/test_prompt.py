import pytest
from src.prompts.system_user_prompt import PromptHandler

analyzer = PromptHandler()

def test_positive_sentiment():
    text = "I absolutely love this new AI technology, it's amazing!"
    result = analyzer.analyze_text(text)
    assert "positive" in result.lower()

def test_negative_sentiment():
    text = "This app is terrible and keeps crashing."
    result = analyzer.analyze_text(text)
    assert "negative" in result.lower()

def test_neutral_summary():
    text = "Python is a programming language created by Guido van Rossum in 1991. It is widely used today."
    result = analyzer.analyze_text(text)
    assert "summary" in result.lower() or "neutral" in result.lower()
