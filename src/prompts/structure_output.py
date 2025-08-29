import openai
import os
from dotenv import load_dotenv
import json

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class StructuredPromptHandler:
    def __init__(self):
        self.model = "gpt-3.5-turbo"

    def analyze_text(self, user_input: str):
        """
        Returns structured JSON output for text analysis.
        """
        system_prompt = (
            "You are an intelligent text analysis system. "
            "Always return your output in valid JSON format with the following keys:\n"
            "{\n"
            "  'sentiment': 'positive | negative | neutral',\n"
            "  'summary': 'string',\n"
            "  'category': 'string',\n"
            "  'confidence': float (0-1)\n"
            "}"
        )

        user_prompt = f"Analyze the following text:\n{user_input}"

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        # Ensure valid JSON output
        content = response["choices"][0]["message"]["content"]
        try:
            structured_output = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Model did not return valid JSON.")

        return structured_output
