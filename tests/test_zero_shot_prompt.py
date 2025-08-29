import pytest
from src.prompts.zero_shot_prompt import ZeroShotPromptHandler

analyzer = ZeroShotPromptHandler()

def test_positive_sentiment():
    text = "I love using AI, it's making my work so much easier!"
    result = analyzer.classify_sentiment(text)
    assert result["sentiment"] == "positive"

def test_negative_sentiment():
    text = "This application is awful, it keeps crashing."
    result = analyzer.classify_sentiment(text)
    assert result["sentiment"] == "negative"

def test_neutral_sentiment():
    text = "The book was published in 2020 by an American author."
    result = analyzer.classify_sentiment(text)
    assert result["sentiment"] == "neutral"
