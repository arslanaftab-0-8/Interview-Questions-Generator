import os
import tempfile
import streamlit as st

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA

# --- 1. Application Configuration and Styling ---

def apply_custom_styling():
    """
    Applies custom CSS for a sophisticated and clean user interface.
    """
    st.markdown("""
        <style>
            /* Main app background */
            .stApp {
                background-image: linear-gradient(to right top, #1d2b64, #f8cdda);
                background-attachment: fixed;
                background-size: cover;
                color: #ffffff;
            }
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: rgba(29, 43, 100, 0.8);
            }
            /* Expander styling */
            .st-expander {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .st-expander header {
                color: #ffffff;
            }
            /* General text and headers */
            h1, h2, h3, p, label {
                color: #ffffff;
            }
            /* Buttons */
            .stButton>button {
                background-color: rgba(248, 205, 218, 0.8);
                color: #1d2b64;
                font-weight: bold;
                border-radius: 8px;
                border: 1px solid rgba(29, 43, 100, 0.3);
            }
            /* Info/Success boxes */
            [data-testid="stInfo"], [data-testid="stSuccess"] {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

# --- 2. Core Logic Functions ---

@st.cache_resource(show_spinner=False)
def initialize_models(_api_key):
    """Initializes and caches the LLM and embedding models."""
    os.environ["GOOGLE_API_KEY"] = _api_key
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, convert_system_message_to_human=True)
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return llm, embedding_model
    except Exception as e:
        st.error(f"🔴 Failed to initialize Google AI models. Please check your API Key. Error: {e}")
        st.stop()

def process_pdf(uploaded_file):
    """Reads and splits the uploaded PDF into processable chunks."""
    with st.spinner("Step 1/4: Processing Document... 📄"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    st.success("Step 1/4: Document processed! ✅")
    return texts

def generate_interview_questions(documents, llm):
    """Generates interview questions based on the document content."""
    with st.spinner("Step 2/4: Generating Interview Questions... 🧠"):
        prompt_template = """
        As an expert technical recruiter, generate 10-15 insightful interview questions based on the provided document. The questions should assess skills, experience, and cultural fit, mixing technical, behavioral, and situational questions.
        Document content: "{text}"
        Provide only the list of questions, each on a new line.
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
        generated_questions_text = chain.run(documents)
    st.success("Step 2/4: Questions generated! ✅")
    return generated_questions_text

def create_qa_chain(texts, llm, embedding_model):
    """Builds a retrieval-based QA chain using a FAISS vector store."""
    with st.spinner("Step 3/4: Building Knowledge Base... 📚"):
        vector_store = FAISS.from_documents(texts, embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )
    st.success("Step 3/4: Knowledge base ready! ✅")
    return qa_chain

def answer_generated_questions(qa_chain, questions_text):
    """Answers each generated question using the QA chain and displays the results."""
    with st.spinner("Step 4/4: Answering Questions... 💬"):
        questions_list = [q.strip() for q in questions_text.split('\n') if q.strip() and '?' in q]
        if not questions_list:
            st.warning("Could not parse any questions from the generated text.")
            return

        st.subheader("💬 DocuMentor's Answers")
        for i, question in enumerate(questions_list):
            st.markdown("---")
            st.markdown(f"**❓ Question {i + 1}:** {question}")
            try:
                result = qa_chain({"query": question})
                st.info(f"**💡 Answer:** {result['result']}")
            except Exception as e:
                st.error(f"Could not answer question: '{question}'. Error: {e}")
    st.success("Step 4/4: Analysis complete! ✅")

# --- 3. Streamlit Application UI ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="DocuMentor", layout="wide")
    apply_custom_styling()

    # --- Sidebar for inputs ---
    with st.sidebar:
        st.header("DocuMentor")
        st.markdown("Your AI-powered document analysis partner.")
        
        api_key = st.text_input("Enter your Google API Key", type="password", key="api_key_input")
        
        if api_key:
            st.markdown("---")
            st.markdown("Upload a PDF (e.g., resume, technical paper) for analysis.")
            uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        else:
            uploaded_file = None

    # --- Main content area ---
    st.title("📄 Intelligent Document Q&A")
    st.markdown("Upload a document, and DocuMentor will generate and answer relevant questions based on its content.")

    if not api_key:
        st.info("Please enter your Google API Key in the sidebar to begin.")
        return

    if uploaded_file:
        llm, embedding_model = initialize_models(api_key)
        
        # Main processing pipeline
        texts = process_pdf(uploaded_file)
        if texts:
            generated_questions = generate_interview_questions(texts, llm)
            with st.expander("🧠 View Generated Interview Questions", expanded=True):
                st.markdown(generated_questions.replace("\n", "\n\n"))

            qa_chain = create_qa_chain(texts, llm, embedding_model)
            answer_generated_questions(qa_chain, generated_questions)
    else:
        st.info("👋 Welcome! Please upload a PDF file to start the analysis.")

if __name__ == "__main__":
    main()
