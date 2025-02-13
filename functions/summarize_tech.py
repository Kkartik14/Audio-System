from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

def summarize_tech(text):
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    client = Groq(api_key=groq_api_key)
    tech_prompt = f"""Please analyze the conversation and extract only technology project-related topics and discussions:

Full Transcript:
{text}

Focus on topics such as:
- Software development projects
- Technical implementations
- System architecture
- Project planning and sprints
- Technical challenges and solutions
- Development tools and technologies
- Technical requirements and specifications

Only include technology project-related content. If no tech projects were discussed, state that clearly.
Pickup Keywords from the conversation and print them seperately as Keywords.

Summary:"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": tech_prompt,
                }
            ],
            model="mixtral-8x7b-32768",  
            temperature=0.3,
            max_tokens=500
        )

        summary = chat_completion.choices[0].message.content.strip()
        print(f"Tech Summary:\n{summary}")
        return summary

    except Exception as e:
        print(f"Error during summarization: {e}")
        return None
    
if __name__ == "__main__":
    example_text = """
    This is an example conversation. We are discussing various topics,
    including technology, project updates, and a bit about finance.
    The project is progressing well, but we have encountered some
    technical challenges. We are using Python and FastAPI for the backend,
    and React for the frontend. We also talked about the budget for next quarter.
    so we discussed about how finincial issues are the one that makes the backbone of India.
    """
    summarize_tech(example_text)