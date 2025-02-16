
import re
import json
import spacy
import fitz  
import streamlit as st
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load skills from JSON file safely
try:
    with open("skills.json", "r", encoding="utf-8") as f:
        skill_list = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    skill_list = []

# Create a PhraseMatcher object for skills
phrase_matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(skill) for skill in skill_list]
phrase_matcher.add("SKILLS", None, *patterns)

def extract_name(text):
    nlp_text = nlp(text)
    for ent in nlp_text.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Unknown"

def get_email_addresses(text):
    return re.findall(r'[\w\.-]+@[\w\.-]+', text)

def get_mobile_numbers(text):
    return re.findall(r'\b\d{10}\b', text)

def extract_skills(text):
    doc = nlp(text)
    matches = phrase_matcher(doc)
    return list({doc[start:end].text for match_id, start, end in matches})

def extract_experience(text):
    return "Experience details extracted (Placeholder)"

def extract_education(text):
    return "Education details extracted (Placeholder)"

def extract_location(text):
    return "Location details extracted (Placeholder)"

def extract_text(file):
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = "\n".join([page.get_text("text") for page in doc])
        return text if text.strip() else "No text extracted."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def calculate_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

# Streamlit UI
st.set_page_config(page_title="Resume Parser", layout="wide")
st.title("📄 Resume Parser & Job Description Matcher")

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
    }
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 Upload Resume(s)")
    uploaded_resumes = st.file_uploader("Upload one or multiple resumes in PDF format.", type=["pdf"], accept_multiple_files=True, key="resume")

with col2:
    st.subheader("📜 Upload Job Description")
    job_description = st.file_uploader("Upload a job description.", type=["pdf"], key="job")

st.markdown("---")

if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = []

data = st.session_state.parsed_data

if st.button("🔍 Parse Resumes"):
    st.session_state.parsed_data = []  # Clear old data
    for resume in uploaded_resumes:
        resume_text = extract_text(resume)
        if not resume_text:
            st.warning(f"Could not extract text from {resume.name}.")
            continue
        st.session_state.parsed_data.append({
            "name": extract_name(resume_text),
            "email": get_email_addresses(resume_text),
            "phone": get_mobile_numbers(resume_text),
            "skills": extract_skills(resume_text),
            "experience": extract_experience(resume_text),
            "education": extract_education(resume_text),
            "location": extract_location(resume_text),
            "text": resume_text,  # Store text for ranking
            "score": None  # Placeholder for ranking
        })
    data = st.session_state.parsed_data

if data:
    st.subheader("📜 Parsed Resumes")
    for idx, res in enumerate(data):
        with st.container():
            st.markdown(f"### 📝 Resume {idx+1}: {res['name']}")
            st.write(f"📧 Email: {res['email']}")
            st.write(f"📞 Phone: {res['phone']}")
            st.write(f"📍 Location: {res['location']}")
            st.write(f"🎓 Education: {res['education']}")
            st.write(f"💼 Experience: {res['experience']}")
            st.write(f"🛠️ Skills: {', '.join(res['skills'])}")
            st.markdown("---")

if job_description and st.button("📊 Match & Rank"):
    job_desc_text = extract_text(job_description)
    if not job_desc_text:
        st.error("Could not extract text from the job description file.")
    else:
        ranked_results = []
        for res in data:
            if not res["text"]:
                continue
            res["score"] = calculate_similarity(res["text"], job_desc_text)
            ranked_results.append(res)
        
        sorted_results = sorted(ranked_results, key=lambda x: x["score"], reverse=True)
        st.subheader("🏆 Ranked Resumes")
        for idx, res in enumerate(sorted_results):
            with st.container():
                st.markdown(f"### 🥇 Rank {idx+1}: {res['name']} - Score: {res['score']:.2f}")
                st.write(f"📧 Email: {res['email']}")
                st.write(f"📞 Phone: {res['phone']}")
                st.write(f"📍 Location: {res['location']}")
                st.write(f"🎓 Education: {res['education']}")
                st.write(f"💼 Experience: {res['experience']}")
                st.write(f"🛠️ Skills: {', '.join(res['skills'])}")
                st.markdown("---")
