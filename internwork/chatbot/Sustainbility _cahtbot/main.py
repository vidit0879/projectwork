import requests
import os
from dotenv import load_dotenv

# If you want to process PDFs
import PyPDF2

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a sustainability and packaging expert specializing in Life Cycle Assessment (LCA), ESG (Environmental, Social, Governance) reporting, "
    "and materiality analysis for packaging. Answer user questions as an industry authority, using up-to-date standards, real-world examples, and "
    "clear explanations tailored to packaging solutions."
)

def validate_api_key():
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key or set the environment variable.")
        return False
    return True

def ask_groq(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 800
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        raise Exception(f"API request failed: {e}")
    except KeyError as e:
        raise Exception(f"Unexpected API response format: {e}")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Could not read PDF: {e}")
    return text

def summarize_and_benchmark_esg(text):
    """Ask the LLM to summarize and benchmark the ESG report."""
    prompt = (
        "Below is the extracted text from a company's ESG report. "
        "Please provide:\n"
        "1. A concise summary of the company's ESG performance.\n"
        "2. A benchmarking analysis compared to industry standards or leaders (if possible).\n"
        "Here is the ESG report:\n"
        "-----\n"
        f"{text[:3500]}"  # Limit to ~3500 chars due to token limits; adjust as needed
        "\n-----\n"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return ask_groq(messages)

def main():
    if not validate_api_key():
        return

    print("🌱 Sustainability Packaging Chatbot")
    print("=" * 40)
    print("Ask me about LCA, ESG reporting, packaging sustainability, and more!")
    print("Type 'esg <file_path>' to analyze an ESG report.")
    print("Type 'exit', 'quit', or 'q' to end the conversation.\n")

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nThank you for using the Sustainability Chatbot! 🌍")
                break

            if user_input.lower().startswith("esg "):
                # User wants to analyze an ESG report
                _, file_path = user_input.split(" ", 1)
                if not os.path.exists(file_path):
                    print("File not found. Please check the file path and try again.")
                    continue
                print("Extracting text from ESG report...")
                text = extract_text_from_pdf(file_path)
                if not text.strip():
                    print("No text could be extracted from the report.")
                    continue
                print("Summarizing and benchmarking the ESG report. Please wait...\n")
                summary = summarize_and_benchmark_esg(text)
                print("\nBot (ESG Summary & Benchmarking):")
                print(summary)
                continue

            if not user_input:
                print("Please enter a question or type 'exit' to quit.")
                continue

            conversation.append({"role": "user", "content": user_input})

            print("\nBot: ", end="", flush=True)
            answer = ask_groq(conversation)
            print(answer)

            conversation.append({"role": "assistant", "content": answer})

        except KeyboardInterrupt:
            print("\n\nGoodbye! 🌍")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()