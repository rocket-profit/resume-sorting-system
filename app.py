import streamlit as st
import os
import json # Added for handling structured data
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
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
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

st.sidebar.success("Access Granted")

# 4. INITIALIZE MODELS
@st.cache_resource
def init_models():
    api_key = os.environ.get("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model='gemini-embedding-2-preview', 
        google_api_key=api_key
    )
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash-lite', 
        google_api_key=api_key
    )
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
st.title("Get top candidates in 2 minutes without manual screening")
st.markdown("### Powered by Rocket Profit Systems")

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
            
            # Extract text
            all_extracted = [{"name": file.name, "text": extract_text(file)} for file in uploaded_files]
            resume_data = [r for r in all_extracted if r.get("text") and str(r["text"]).strip()]
            
            if len(resume_data) < len(all_extracted):
                st.warning(f"⚠️ {len(all_extracted) - len(resume_data)} file(s) skipped (no readable text).")

            if not resume_data:
                st.error("No readable text found in any uploaded resumes. Process stopped.")
                st.stop()

            # Create Embeddings and calculate similarity
            jd_vector = embeddings.embed_query(jd_text)
            resume_texts = [r["text"] for r in resume_data]
            resume_vectors = embeddings.embed_documents(resume_texts)
            scores = cosine_similarity([jd_vector], resume_vectors)[0]

            # Pair names/texts with their scores and sort descending
            scored_resumes = []
            for i, score in enumerate(scores):
                scored_resumes.append({
                    "name": resume_data[i]["name"],
                    "text": resume_texts[i],
                    "score": score
                })
            scored_resumes = sorted(scored_resumes, key=lambda x: x["score"], reverse=True)

            # --- HYBRID FILTERING LOGIC ---
            MIN_SCORE = 0.50
            TOP_PERCENT = 0.50
            
            max_candidates = max(1, int(len(scored_resumes) * TOP_PERCENT))
            final_candidates = [c for c in scored_resumes[:max_candidates] if c["score"] >= MIN_SCORE]

            if not final_candidates and scored_resumes:
                final_candidates = [scored_resumes[0]]
                st.info("⚠️ No candidates met the strict 50% keyword match threshold. Showing the closest available profile.")

            # --- LLM JSON EXTRACTION LOGIC ---
            results = []
            for candidate in final_candidates:
                prompt = f"""You are an expert HR. Critically evaluate this resume against the JD. 
                Resume: {candidate['text']} 
                JD: {jd_text}
                
                Provide your response STRICTLY in the following JSON format. Do not use markdown blocks, just raw JSON:
                {{
                    "candidate_name": "Full Name from resume",
                    "contact_info": {{
                        "email": "Email address (or N/A)",
                        "phone": "Phone number (or N/A)"
                    }},
                    "analysis": {{
                        "pros": ["bullet 1", "bullet 2", "bullet 3"],
                        "critical_con": "Main missing skill",
                        "verdict": "1-2 line summary"
                    }}
                }}
                """
                analysis = llm.invoke(prompt)
                
                try:
                    # Clean the response to ensure valid JSON parsing
                    clean_content = analysis.content.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean_content)
                    
                    results.append({
                        "Filename": candidate["name"],
                        "Name" : data.get("candidate_name", candidate["name"]),
                        "Email": data.get("contact_info", {}).get("email", "N/A"),
                        "Phone": data.get("contact_info", {}).get("phone", "N/A"),
                        "Similarity" : round(candidate["score"] * 100, 1),
                        "Analysis" : data.get("analysis", {})
                    })
                except Exception as e:
                    # Failsafe in case the LLM breaks formatting
                    st.error(f"Failed to extract structured data for {candidate['name']}. Continuing...")
                    continue
             
            st.success(f"✅ Successfully filtered down to the top {len(results)} candidate(s)!")

            # --- DISPLAY LOGIC ---
            for res in results:
                with st.expander(f"{res['Name']} - Fit Score: {res['Similarity']}%"):
                    col_contact, col_eval = st.columns([1, 2])
                    
                    with col_contact:
                        st.write("**Contact Details:**")
                        st.write(f"📧 {res['Email']}")
                        st.write(f"📞 {res['Phone']}")
                        st.caption(f"Original File: {res['Filename']}")
                    
                    with col_eval:
                        st.write("**Pros:**")
                        for pro in res['Analysis'].get('pros', []):
                            st.write(f"- {pro}")
                        st.write(f"**Critical Con:** {res['Analysis'].get('critical_con', 'N/A')}")
                        st.info(f"**Verdict:** {res['Analysis'].get('verdict', 'N/A')}")
                    
            # --- UPGRADED CSV EXPORT ---
            if results:
                # Flatten the analysis dictionary for cleaner CSV export
                df = pd.DataFrame(results)
                df['Pros'] = df['Analysis'].apply(lambda x: " | ".join(x.get('pros', [])))
                df['Critical Con'] = df['Analysis'].apply(lambda x: x.get('critical_con', ''))
                df['Verdict'] = df['Analysis'].apply(lambda x: x.get('verdict', ''))
                
                # Select only the columns we want to export
                export_df = df[['Name', 'Email', 'Phone', 'Similarity', 'Pros', 'Critical Con', 'Verdict']]
                csv = export_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Full Ranking Report (CSV)",
                    data=csv,
                    file_name='rocket_profit_candidates.csv',
                    mime='text/csv',
                )