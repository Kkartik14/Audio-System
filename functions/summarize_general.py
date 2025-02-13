from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


def summarize_general(text):
    """Generates a general summary of the given text using Groq."""

    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    client = Groq(api_key=groq_api_key)

    prompt = f"""Please provide a comprehensive summary of the entire conversation so far:

Context: This is an ongoing conversation/speech. Provide a cohesive summary that captures the main points discussed from the beginning until now.

Full Transcript:
{text}

Please provide:
1. A concise overall summary
2. Key points discussed
3. Any significant transitions or changes in topic

Summary:"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=500
        )
        summary = chat_completion.choices[0].message.content.strip()
        print(f"General Summary:\n{summary}")
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


if __name__ == "__main__":
    example_text = """
    This is an example conversation. We are discussing various topics,
    including technology, project updates, and a bit about finance.
    The project is progressing well, but we have encountered some
    technical challenges.  We also talked about the budget for next quarter.
    """
    summarize_general(example_text)