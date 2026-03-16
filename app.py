import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import docx

# 1. SETUP & PAGE CONFIG (Must be first)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    st.set_page_config(page_title='Rocket Profit AI', page_icon="🚀", layout="wide", initial_sidebar_state="collapsed")
else:
    st.set_page_config(page_title='Rocket Profit AI', page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# 2. LOAD SECRETS (The Bridge for Cloud Deployment)
load_dotenv() 

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "APP_PASSWORD" in st.secrets:
        os.environ["APP_PASSWORD"] = st.secrets["APP_PASSWORD"]
except FileNotFoundError:
    pass 

Secret_pass = os.getenv("APP_PASSWORD")

# 3. THE GATEKEEPER (Absolute Security Lock)
if not st.session_state.authenticated:
    with st.sidebar:
        st.header("Security Gateway")
        user_pass = st.text_input("Enter Access Key", type="password")
        
        if user_pass == Secret_pass and Secret_pass is not None:
            st.session_state.authenticated = True
            st.rerun()  # Instantly reloads app to collapse sidebar
        elif user_pass != "":
            st.error("Access Denied")
            
    st.info("🔒 Please enter your access key in the sidebar to unlock the AI Sorter.")
    st.stop() # CRITICAL: This completely stops the app from rendering the UI below!

# --- MAIN APP RUNS ONLY IF AUTHENTICATED ---
st.sidebar.success("Access Granted: A1HR Consulting")

# 4. INITIALIZE MODELS (Hardwired for Cloud Stability)
@st.cache_resource
def init_models():
    # Try to get the key from the environment, and if that fails, grab it directly from Streamlit secrets
    api_key = os.environ.get("GOOGLE_API_KEY") 
    
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            api_key = None
            
    # The Safety Net
    if not api_key:
        st.error("🚨 CRITICAL SYSTEM ERROR: API Key missing from Streamlit Secrets.")
        st.stop()
        
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=api_key)
    return embeddings, llm

embeddings, llm = init_models()
# 5. HELPER FUNCTION
def extract_text(feed):
    text = ""
    file_extension = feed.name.split('.')[-1].lower()
    try:
        if file_extension == 'pdf':
            pdf_reader = PdfReader(feed)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        elif file_extension == 'docx':
            doc = docx.Document(feed)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_extension == 'txt':
            text = feed.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading {feed.name}: {e}")
    return text

# 6. UI LAYOUT
st.title("🚀 Rocket Profit: Stop sorting resumes manually.")
st.markdown("### Revamping Operational Efficiency for HR Consulting")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    jd_text = st.text_area("Paste the Job Description here...", height=250)
    
with col2:
    st.subheader("2. Upload Resumes")
    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
# 7. PROCESSING SECTION
if st.button("Start Sorting"):
    if not jd_text or not uploaded_files:
        st.warning("Please Provide both a Job Description and at least 1 Resume")
    else:
        with st.spinner("Analyzing and Ranking Candidates..."):
            all_extracted = [{"name": file.name, "text": extract_text(file)} for file in uploaded_files]
            resume_data = [r for r in all_extracted if r.get("text") and str(r["text"]).strip()]
            
            if len(resume_data) < len(all_extracted):
                st.warning(f"⚠️ {len(all_extracted) - len(resume_data)} file(s) were skipped because no readable text was found.")

            if not resume_data:
                st.error("No readable text found in any uploaded resumes. Process stopped.")
                st.stop()

            jd_vector = embeddings.embed_query(jd_text)
            resume_texts = [r["text"] for r in resume_data]
            resume_vectors = embeddings.embed_documents(resume_texts)

            scores = cosine_similarity([jd_vector], resume_vectors)[0]

            results = []
            for i, score in enumerate(scores):
                if score > 0.3:
                    prompt = f"""You are an expert HR consultant. Critically evaluate this resume against the JD. 
                    Resume: {resume_texts[i]} 
                    JD: {jd_text}
                    
                    Provide a highly structured response:
                    - 3 Pros: (Bullet points)
                    - 1 Critical Con/Missing Skill:
                    - Overall Verdict: (1-2 line summary)
                    """
                    analysis = llm.invoke(prompt)
                    results.append({
                        "Name" : resume_data[i]["name"],
                        "Similarity" : round(score * 100, 1),
                        "AI Analysis" : analysis.content
                    })
            
            results = sorted(results, key=lambda x: x["Similarity"], reverse=True)
            st.success(f"✅ Successfully analyzed {len(uploaded_files)} resumes!")

            for res in results:
                with st.expander(f"{res['Name']} - Semantic Match: {res['Similarity']}%"):
                    st.markdown(res["AI Analysis"])
                    
            if results:
                df = pd.DataFrame(results)
                csv = df[['Name', 'Similarity']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Ranking Report (CSV)",
                    data=csv,
                    file_name='resume_sorting.csv',
                    mime='text/csv',
                )