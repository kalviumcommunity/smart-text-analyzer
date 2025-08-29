# prompts/multi_shot_prompting.py

from openai import OpenAI

client = OpenAI()

def multi_shot_prompting(input_query: str):
    """
    Multi-shot prompting example:
    Multiple examples are provided before the final query.
    This ensures consistent, correct, and structured output.
    """

    prompt = """
You are an assistant that provides structured information about programming languages.
Return the output in JSON format with fields: { "language": string, "creator": string, "year": number }.

Examples:
Input: "Tell me about Python"
Output: { "language": "Python", "creator": "Guido van Rossum", "year": 1991 }

Input: "Tell me about JavaScript"
Output: { "language": "JavaScript", "creator": "Brendan Eich", "year": 1995 }

Input: "Tell me about Java"
Output: { "language": "Java", "creator": "James Gosling", "year": 1995 }

Now, answer this:
Input: "{query}"
""".format(query=input_query)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content
