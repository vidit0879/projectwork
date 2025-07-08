import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3-70b-instruct"

SYSTEM_PROMPT = (
    "You are a sustainability and packaging expert specializing in Life Cycle Assessment (LCA), ESG (Environmental, Social, Governance) reporting, "
    "and materiality analysis for packaging. Answer user questions as an industry authority, using up-to-date standards, real-world examples, and "
    "clear explanations tailored to packaging solutions."
)

def validate_api_key():
    """Validate that the API key is available."""
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key or set the environment variable.")
        return False
    return True

def ask_groq(messages):
    """Send request to Groq API and return response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 800
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    except KeyError as e:
        raise Exception(f"Unexpected API response format: {e}")

def main():
    """Main chatbot loop."""
    if not validate_api_key():
        return
    
    print("🌱 Sustainability Packaging Chatbot")
    print("=" * 40)
    print("Ask me about LCA, ESG reporting, packaging sustainability, and more!")
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
