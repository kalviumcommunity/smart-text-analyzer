from openai import OpenAI
import tiktoken  # For token counting (ensure installed: pip install tiktoken)

# Initialize OpenAI client
client = OpenAI()

# Function to count tokens using tiktoken
def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def run_tokenization_demo():
    # Prompt (user input)
    user_prompt = "Explain the importance of clean code in software development."

    # Count input tokens
    input_tokens = count_tokens(user_prompt, "gpt-3.5-turbo")

    # Make AI call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI tutor."},
            {"role": "user", "content": user_prompt}
        ]
    )

    output_text = response.choices[0].message.content

    # Count output tokens
    output_tokens = count_tokens(output_text, "gpt-3.5-turbo")

    # Log everything
    print("=== Tokenization Log ===")
    print(f"User Prompt: {user_prompt}")
    print(f"Input Tokens: {input_tokens}")
    print(f"AI Response: {output_text}")
    print(f"Output Tokens: {output_tokens}")
    print(f"Total Tokens Used: {input_tokens + output_tokens}")

    return output_text

if __name__ == "__main__":
    run_tokenization_demo()
