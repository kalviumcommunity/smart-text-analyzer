import pytest
from src.prompts.dynamic_prompt import DynamicPromptHandler

analyzer = DynamicPromptHandler()

def test_short_text_sentiment():
    text = "I love AI!"
    result = analyzer.analyze_text(text)
    assert "sentiment" in result
    assert 0 <= result["confidence"] <= 1

def test_long_text_summary():
    text = "Artificial intelligence is transforming industries worldwide. It improves efficiency, enhances customer experience, and creates new opportunities in healthcare, finance, and education."
    result = analyzer.analyze_text(text)
    assert "summary" in result
    assert "sentiment" in result
    assert isinstance(result.get("key_points", []), list)

def test_dynamic_switching():
    short_text = "Bad service!"
    long_text = "Python is a versatile programming language. It is widely used for AI, data science, and automation."
    short_result = analyzer.analyze_text(short_text)
    long_result = analyzer.analyze_text(long_text)
    assert "sentiment" in short_result
    assert "summary" in long_result
