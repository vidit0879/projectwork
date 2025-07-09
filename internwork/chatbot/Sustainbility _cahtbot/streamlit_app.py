import streamlit as st
import requests
import os
from dotenv import load_dotenv
import PyPDF2
import io

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a sustainability and packaging expert specializing in Life Cycle Assessment (LCA), ESG (Environmental, Social, Governance) reporting, "
    "and materiality analysis for packaging. Answer user questions as an industry authority, using up-to-date standards, real-world examples, and "
    "clear explanations tailored to packaging solutions."
)

def ask_groq(messages):
    """Send request to Groq API and return response."""
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
            st.error(f"API Error {e.response.status_code}: {e.response.text}")
        else:
            st.error(f"Request failed: {e}")
        return None
    except KeyError as e:
        st.error(f"Unexpected API response format: {e}")
        return None

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
    return text

def summarize_and_benchmark_esg(text):
    """Ask the LLM to summarize and benchmark the ESG report."""
    prompt = (
        "Below is the extracted text from a company's ESG report. "
        "Please provide:\n"
        "1. A concise summary of the company's ESG performance.\n"
        "2. A benchmarking analysis compared to industry standards or leaders (if possible).\n"
        "3. Key recommendations for improvement.\n"
        "Here is the ESG report:\n"
        "-----\n"
        f"{text[:3500]}"  # Limit to ~3500 chars due to token limits
        "\n-----\n"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return ask_groq(messages)

def ai_sustainability_assessment(material, weight, recyclable, renewable):
    prompt = (
        f"Packaging parameters:\n"
        f"- Material: {material}\n"
        f"- Weight: {weight} grams\n"
        f"- Recyclable: {'Yes' if recyclable else 'No'}\n"
        f"- Made from renewable resources: {'Yes' if renewable else 'No'}\n\n"
        "Based on these, provide:\n"
        "1. A sustainability score out of 10 (with justification).\n"
        "2. A brief assessment and recommendations for improvement."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return ask_groq(messages)

def main():
    st.set_page_config(
        page_title="Sustainability Packaging Chatbot",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("🌱 Sustainability Packaging Chatbot")
    st.markdown("Expert advice on LCA, ESG reporting, packaging sustainability, and materiality analysis")
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 ESG Report Analysis", "♻️ AI Sustainability Score Calculator"])
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot specializes in:
        - Life Cycle Assessment (LCA)
        - ESG reporting
        - Packaging sustainability
        - Materiality analysis
        - Environmental standards
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Check API key
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Please add it to your .env file.")
        st.stop()
    
    # Chat Tab
    with tab1:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about sustainability, packaging, LCA, or ESG..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Prepare conversation for API
            conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
            conversation.extend(st.session_state.messages)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ask_groq(conversation)
                    
                    if response:
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Failed to get response. Please try again.")
    
    # ESG Report Analysis Tab
    with tab2:
        st.header("📄 ESG Report Analysis")
        st.markdown("Upload an ESG report PDF for analysis and benchmarking")
        
        uploaded_file = st.file_uploader(
            "Choose an ESG report PDF file", 
            type="pdf",
            help="Upload a PDF file containing the ESG report you want to analyze"
        )
        
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            
            if st.button("Analyze ESG Report", type="primary"):
                with st.spinner("Extracting text from PDF..."):
                    text = extract_text_from_pdf(uploaded_file)
                
                if text.strip():
                    with st.spinner("Analyzing ESG report and generating benchmarking analysis..."):
                        analysis = summarize_and_benchmark_esg(text)
                    
                    if analysis:
                        st.subheader("📊 ESG Analysis Results")
                        st.markdown(analysis)
                        
                        # Option to download analysis
                        st.download_button(
                            label="Download Analysis",
                            data=analysis,
                            file_name=f"esg_analysis_{uploaded_file.name.replace('.pdf', '')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Failed to analyze the ESG report. Please try again.")
                else:
                    st.error("No text could be extracted from the PDF. Please check if the file is readable.")

    # AI Sustainability Score Calculator Tab
    with tab3:
        st.header("♻️ AI Sustainability Score Calculator")
        st.markdown("Input your packaging parameters to get an AI-powered sustainability score and assessment.")

        material = st.selectbox(
            "Material",
            ["Plastic", "Glass", "Aluminum", "Paper", "Bioplastic", "Compostable", "Other"]
        )
        weight = st.number_input("Weight (grams)", min_value=0.0, step=0.1)
        recyclable = st.radio("Is it recyclable?", ["Yes", "No"]) == "Yes"
        renewable = st.radio("Is it made from renewable resources?", ["Yes", "No"]) == "Yes"

        if st.button("Analyze with AI"):
            with st.spinner("Analyzing with AI..."):
                ai_result = ai_sustainability_assessment(material, weight, recyclable, renewable)
            if ai_result:
                st.subheader("🤖 AI Sustainability Assessment")
                st.markdown(ai_result)
            else:
                st.error("Failed to get AI analysis. Please try again.")

if __name__ == "__main__":
    main()