import pytest
from prompts.multi_shot_prompting import multi_shot_prompting
import json

def test_multi_shot_prompting():
    # Example input
    query = "Tell me about C++"
    
    result = multi_shot_prompting(query)

    # Try to parse JSON
    try:
        data = json.loads(result.replace("'", '"'))
    except Exception:
        pytest.fail("Output is not valid JSON")

    # Check required keys
    assert "language" in data
    assert "creator" in data
    assert "year" in data
    assert data["language"] == "C++"
