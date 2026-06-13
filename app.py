import os
import re
import csv
import base64
import random
import datetime

import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_tags import st_tags

import nltk
from nltk.corpus import stopwords

# --------------- EXTERNAL COURSES / VIDEOS ---------------
from Courses import (
    ds_course, web_course, android_course,
    ios_course, uiux_course, resume_videos, interview_videos
)

# --------------- BASIC NLP SETUP ---------------
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# --------------- PAGE CONFIG ---------------
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

LOGO_PATH = "Vision.jpg"
UPLOAD_DIR = "Uploaded_Resumes"
SKILLS_FILE = "skills.csv"

os.makedirs(UPLOAD_DIR, exist_ok=True)
if not os.path.exists(SKILLS_FILE):
    open(SKILLS_FILE, "w").close()

# --------------- LOAD SKILLS CSV ---------------
skills_list = []
with open(SKILLS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        skills_list.extend(row)

# --------------- EXTRA TECH KEYWORDS ---------------
TECH_KEYWORDS = [
    # languages
    "python", "java", "c++", "c#", "javascript", "typescript", "kotlin",
    "swift", "go", "ruby", "php", "r", "scala", "dart",
    # web
    "html", "css", "react", "angular", "vue", "django", "flask", "spring",
    "node", "node.js", "express", "rest api", "graphql",
    # data / ml
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
    "keras", "machine learning", "deep learning", "sql", "mysql",
    "postgresql", "mongodb", "power bi", "tableau",
    # mobile
    "android", "android studio", "jetpack", "ios", "xcode", "objective-c",
    "cocoa", "swiftui", "flutter", "react native",
    # devops / cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "github",
    "gitlab", "ci/cd",
    # ui/ux & design
    "figma", "adobe xd", "photoshop", "illustrator", "indesign",
    "wireframe", "prototype", "prototyping", "user research",
    # misc
    "linux", "bash", "shell scripting", "oop", "data structures", "algorithms"
]

# --------------- DOMAIN RULES ---------------
DOMAIN_KEYWORDS = {
    "Data Science": [
        "python", "pandas", "numpy", "machine learning", "ml",
        "deep learning", "tensorflow", "pytorch", "scikit-learn", "statistics"
    ],
    "Web Development": [
        "html", "css", "javascript", "react", "angular", "vue",
        "django", "flask", "node", "node.js", "php"
    ],
    "Android Development": [
        "android", "kotlin", "java", "android studio", "jetpack", "flutter"
    ],
    "iOS Development": [
        "ios", "swift", "xcode", "objective-c", "cocoa", "swiftui"
    ],
    "UI-UX Development": [
        "ux", "ui", "figma", "adobe xd", "wireframe", "prototype",
        "prototyping", "user research"
    ],
    "Python Development": [
    "Python","SQL","OOP","Data Structures","Algorithms","Git","GitHub",
    "Streamlit","File Handling","Debugging"
    ]
}

DOMAIN_RECOMMENDED_SKILLS = {
    "Data Science": [
        "Python", "Pandas", "NumPy", "Statistics",
        "Machine Learning", "Deep Learning",
        "Scikit-learn", "TensorFlow or PyTorch",
        "SQL", "Data Visualization (Matplotlib/Seaborn/Power BI/Tableau)"
    ],
    "Web Development": [
        "HTML5", "CSS3", "JavaScript (ES6+)",
        "React / Angular / Vue",
        "Backend (Node.js/Express or Django/Flask)",
        "REST APIs", "Databases (SQL/NoSQL)",
        "Responsive Design", "Git & GitHub", "Cloud Deployment"
    ],
    "Android Development": [
        "Kotlin", "Android SDK", "Android Jetpack",
        "Material Design", "REST APIs", "Room DB",
        "Unit Testing", "Play Store Deployment"
    ],
    "iOS Development": [
        "Swift", "SwiftUI or UIKit", "Xcode",
        "Cocoa Touch", "REST APIs/JSON",
        "Core Data", "Unit/UI Testing", "App Store Deployment"
    ],
    "UI-UX Development": [
        "User Research", "Information Architecture",
        "Wireframing", "Prototyping",
        "Figma or Adobe XD", "Design Systems",
        "Interaction Design", "Usability Testing"
    ],
    "Python Development": [
    "Python",
    "SQL",
    "OOP",
    "Data Structures",
    "Algorithms",
    "Git",
    "GitHub",
    "Streamlit",
    "File Handling",
    "Debugging"
]
}

# --------------- DB CONNECTION (LOCAL / SIMPLE) ---------------
def insert_data(
    name,
    email,
    res_score,
    timestamp,
    no_of_pages,
    reco_field,
    cand_level,
    skills,
    recommended_skills,
    courses
):

    file_name = "user_data.csv"

    row = {
        "Name": name,
        "Email": email,
        "Resume Score": res_score,
        "Timestamp": timestamp,
        "Pages": no_of_pages,
        "Field": reco_field,
        "Level": cand_level,
        "Skills": skills,
        "Recommended Skills": recommended_skills,
        "Courses": courses
    }

    df = pd.DataFrame([row])

    if os.path.exists(file_name):
        df.to_csv(file_name, mode="a", header=False, index=False)
    else:
        df.to_csv(file_name, index=False)
# --------------- PDF VIEW/TEXT ---------------
def show_pdf(file_path: str):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    iframe = (
        f"<iframe src='data:application/pdf;base64,{b64}' "
        f"width='700' height='900' type='application/pdf'></iframe>"
    )
    st.markdown(iframe, unsafe_allow_html=True)


def get_resume_text(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception as e:
        print("pdfplumber error:", e)

    if len(text.strip()) < 50:
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        except Exception as e:
            print("fitz error:", e)

    return text

# --------------- ✅ ROBUST NAME DETECTION ---------------
def extract_name(text):

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines[:10]:

        # Skip email lines
        if "@" in line:
            continue

        # Skip LinkedIn/GitHub lines
        if "linkedin" in line.lower():
            continue

        if "github" in line.lower():
            continue

        # Skip phone number lines
        if re.search(r"\d{10}", line):
            continue

        # Skip job titles
        skip_words = [
            "developer",
            "engineer",
            "student",
            "aspiring",
            "python",
            "software"
        ]

        if any(word in line.lower() for word in skip_words):
            continue

        cleaned = re.sub(r"[^A-Za-z.\s]", "", line).strip()

        if len(cleaned) >= 5:
            return cleaned.title()

    return "Not Found"

def extract_email(text):

    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    matches = re.findall(pattern, text)

    if matches:
        return matches[0]

    return "Not Found"


def extract_phone(text):

    text = text.replace("\n", " ")

    pattern = r"(?:\+91[\-\s]?)?[6-9]\d{9}"

    matches = re.findall(pattern, text)

    if matches:
        return matches[0]

    pattern2 = r"\+?\d[\d\s\-\(\)]{8,15}\d"

    matches = re.findall(pattern2, text)

    if matches:
        return matches[0]

    return "Not Found"

# --------------- SKILLS (multi-stage) ---------------
def extract_skills_from_text(text, skills_list):

    text_lower = text.lower()

    detected = set()

    # Skills from skills.csv
    for skill in skills_list:

        skill = skill.strip()

        if not skill:
            continue

        if skill.lower() in text_lower:
            detected.add(skill.title())

    # Skills from TECH_KEYWORDS
    for skill in TECH_KEYWORDS:
        if len(skill) < 3:
            continue
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_lower):
            detected.add(skill.title())

    return sorted(list(detected))

# --------------- ATS KEYWORDS & MATCH ---------------
def extract_keywords_for_ats(text: str):
    text = re.sub(r"[^a-zA-Z0-9+.# ]", " ", text.lower())
    tokens = text.split()
    kws = []
    for t in tokens:
        if len(t) < 3:
            continue
        if t in STOPWORDS:
            continue
        if t.isdigit():
            continue
        kws.append(t)
    return set(kws)


def ats_match(resume_text: str, job_text: str):
    job_kw = extract_keywords_for_ats(job_text)
    res_kw = extract_keywords_for_ats(resume_text)
    if not job_kw:
        return 0.0, set(), set()
    matched = job_kw & res_kw
    missing = job_kw - res_kw
    score = round(len(matched) / len(job_kw) * 100, 2)
    return score, matched, missing

# --------------- SECTION DETECTION ---------------
def detect_sections(text: str):
    patterns = {
        "Summary / Objective": r"(summary|objective)",
        "Skills": r"\bskills\b",
        "Experience": r"(experience|work history|employment)",
        "Education": r"education",
        "Projects": r"projects?",
        "Certifications": r"certifications?",
        "Achievements": r"(achievements?|awards?)",
    }
    presence = {}
    for name, pat in patterns.items():
        presence[name] = bool(re.search(pat, text, re.I))
    return presence

# --------------- FIELD PREDICTION & RECO ---------------
def predict_field_and_reco(extracted_skills, full_text_lower):
    sl = [s.lower() for s in extracted_skills]
    scores = {}

    def count(keywords):
        return sum(
        1 for k in keywords
        if k.lower() in [s.lower() for s in extracted_skills]
    )

    for domain, kws in DOMAIN_KEYWORDS.items():
        scores[domain] = count(kws)

    best_domain, best_score = "", 0
    for d, s in scores.items():
        if s > best_score:
            best_domain, best_score = d, s

    if best_score == 0:
        return "", [], []

    rec_skills = DOMAIN_RECOMMENDED_SKILLS.get(best_domain, [])
    if best_domain == "Data Science":
        course_list = ds_course
    elif best_domain == "Web Development":
        course_list = web_course
    elif best_domain == "Android Development":
        course_list = android_course
    elif best_domain == "iOS Development":
        course_list = ios_course
    elif best_domain == "UI-UX Development":
        course_list = uiux_course
    elif best_domain == "Python Development":
        course_list = []
    else:
        course_list = []

    return best_domain, rec_skills, course_list

# --------------- COURSE RECOMMENDER ---------------
def course_recommender(course_list):
    st.subheader("*Courses & Certificates 🎓 Recommendations*")
    if not course_list:
        st.info("No course list configured.")
        return []

    random.shuffle(course_list)
    max_reco = min(10, len(course_list))
    default = min(4, max_reco)
    n = st.slider("Choose Number of Course Recommendations:", 1, max_reco, default)
    chosen = []
    for i, (name, link) in enumerate(course_list[:n], start=1):
        st.markdown(f"({i}) [{name}]({link})")
        chosen.append(name)
    return chosen

# --------------- RESUME SCORE ---------------
def calculate_resume_score(resume_data, recommended_skills):

    score = 0
    matched = 0
    user_skills = resume_data.get("skills", [])

    # Case-insensitive comparison
    user_set = {skill.lower().strip() for skill in user_skills}
    reco_set = {skill.lower().strip() for skill in recommended_skills}

    if reco_set:
        matched = len(user_set.intersection(reco_set))
        score += (matched / len(reco_set)) * 60

    # Page score
    pages = resume_data.get("no_of_pages", 1)

    if pages == 1:
        score += 15
    elif pages == 2:
        score += 10
    else:
        score += 5

    # Basic completeness
    completeness = 0

    if resume_data.get("name"):
        completeness += 1

    if resume_data.get("email"):
        completeness += 1

    if resume_data.get("mobile_number"):
        completeness += 1

    if user_skills:
        completeness += 1

    score += (completeness / 4) * 25
    st.write("User Skills:", user_skills)
    st.write("Recommended Skills:", recommended_skills)
    st.write("Matched Skills:", matched)
    st.write("Current Score:", score)
    return round(min(score, 100), 2)
# --------------- TIPS ---------------
def generate_resume_tips(res_score, ats_score, section_presence, no_of_pages,
                         level, missing_kw, extracted_skills,
                         recommended_skills, is_scanned):
    tips = []

    if is_scanned:
        tips.append(
            "Your PDF looks like a scanned/image resume. Convert it to a text-based PDF "
            "so ATS and recruiters can search your content."
        )

    if not section_presence.get("Summary / Objective", False):
        tips.append("Add a 2–3 line Summary/Objective at the top focused on your target role.")

    if not section_presence.get("Skills", False) or len(extracted_skills) < 5:
        tips.append(
            "Create a dedicated **Skills** section with 8–15 relevant skills "
            "(languages, frameworks, tools) in a clean list."
        )

    if level == "Fresher" and not section_presence.get("Projects", False):
        tips.append(
            "As a fresher, add 2–4 **Projects** with tech stack, your role, and results "
            "to prove your practical skills."
        )

    if level != "Fresher" and not section_presence.get("Experience", False):
        tips.append(
            "Add a **Work Experience** section in reverse chronological order "
            "with achievement-focused bullet points."
        )

    if no_of_pages > 2:
        tips.append("Try to keep the resume within **1–2 pages**. Remove old or irrelevant details.")

    if len(extracted_skills) > 18:
        tips.append(
            "You listed many skills; group them into categories "
            "(Languages, Frameworks, Tools, Cloud, etc.) to improve readability."
        )

    if res_score < 60:
        tips.append(
            "Overall structure needs work: use clear headings, consistent bullets, and avoid long paragraphs."
        )
    elif res_score < 80:
        tips.append(
            "Resume is okay, but improve by quantifying impact in bullets "
            "(% improvement, time saved, users, revenue, etc.)."
        )
    else:
        tips.append("Resume is strong; keep tailoring it for each specific job.")

    if ats_score is not None:
        if ats_score < 50:
            tips.append(
                "ATS match is low. Re-read the Job Description and add the **exact keywords** "
                "in your Skills and Experience (only for skills you actually have)."
            )
        elif ats_score < 75:
            tips.append(
                "ATS match is moderate. Sprinkle more JD keywords into your bullets "
                "and project descriptions naturally."
            )

        if missing_kw:
            top_missing = list(sorted(missing_kw))[:10]
            tips.append(
                "Consider genuinely learning or highlighting these missing JD keywords "
                "if they are relevant for you: " + ", ".join(top_missing) + "."
            )

    if recommended_skills:
        tips.append(
            "Build or strengthen the **Recommended Skills** shown above; "
            "they’re critical for your predicted field."
        )

    return tips

# --------------- MAIN APP ---------------
def run():
    st.sidebar.markdown("## Choose User")
    mode = st.sidebar.selectbox("Choose among the given options:", ["Normal User", "Admin"])

    # Logo
    try:
        img = Image.open(LOGO_PATH)
        img = img.resize((250, 250))
        st.image(img)
    except Exception:
        st.title("Smart Resume Analyzer")

    # ---------- NORMAL USER ----------
    if mode == "Normal User":
        st.markdown("## Smart Resume Analyzer")

        pdf_file = st.file_uploader("Upload Your Resume", type=["pdf"])
        if not pdf_file:
            return

        save_path = os.path.join(UPLOAD_DIR, pdf_file.name)
        with open(save_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        st.markdown("### Uploaded Resume Preview")
        show_pdf(save_path)

        text = get_resume_text(save_path)
        is_scanned = len(text.strip()) < 120
        if is_scanned:
            st.error(
                "Your resume seems to have very little extractable text. "
                "It may be an image-based or scanned PDF. "
                "ATS and skills analysis might be inaccurate."
            )

        extracted_skills = extract_skills_from_text(text, skills_list)
        name_raw = extract_name(text)
        email_raw = extract_email(text)
        phone_raw = extract_phone(text)
        no_of_pages = len(fitz.open(save_path))
        section_presence = detect_sections(text)

        st.markdown("---")
        st.markdown("## Resume Analysis")

        st.success(f"Hello {name_raw} 👋")

        name = st.text_input("Name", value=name_raw)
        email = st.text_input("Email", value=email_raw)
        mobile_number = st.text_input("Contact", value=phone_raw)

        st.write(f"**Pages:** {no_of_pages}")

        if no_of_pages == 1:
            level = "Fresher"
        elif no_of_pages == 2:
            level = "Intermediate"
        else:
            level = "Experienced"
        st.markdown(f"### 🧠 Candidate Level: **{level}**")

        st.markdown("### Skills Detected 👍")
        extracted_skills = st_tags(
            label="### Skills You Have",
            value=extracted_skills,
            key="skills_user",
        )

        resume_data = {
            "skills": extracted_skills,
            "name": name,
            "email": email,
            "mobile_number": mobile_number,
            "no_of_pages": no_of_pages,
        }

        # field prediction
        field, recommended_skills, course_list = predict_field_and_reco(
            extracted_skills, text.lower()
        )
        rec_course = []

        if field:
            st.markdown(f"### 🔮 Predicted Field: **{field}**")
            if recommended_skills:
                st_tags(
                    label="### Recommended Skills",
                    value=recommended_skills,
                    key="skills_reco",
                )
            if course_list:
                rec_course = course_recommender(course_list)
        else:
            st.info(
                "Could not confidently predict a specific field. "
                "Make sure your main technologies are clearly listed."
            )
            st_tags(label="### Recommended Skills", value=[], key="skills_reco")

        res_score = calculate_resume_score(resume_data, recommended_skills)
        st.markdown(f"## 📊 Resume Score: **{res_score}/100**")
        if res_score > 80:
            st.success("Excellent! Resume is well structured and fairly optimized.")
        elif res_score > 60:
            st.info("Good resume. You can still polish structure and add stronger impact.")
        else:
            st.warning("Weak resume. Work on format, clarity, and relevant keywords.")

        # ATS
        st.markdown("---")
        st.markdown("## ATS Analysis (Job Match)")

        job_title = st.text_input("Target Job Title (optional)", value=field or "")
        job_desc = st.text_area(
            "Paste Job Description / Required Skills here (for ATS match):",
            height=150,
            help="Copy a job description, paste here, and see how well your resume matches it.",
        )

        ats_score, matched_kw, missing_kw = None, set(), set()
        if job_desc.strip():
            ats_score, matched_kw, missing_kw = ats_match(text, job_desc)
            st.markdown(f"### ✅ ATS Match Score: **{ats_score}%**")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Matched Keywords")
                st.write(", ".join(sorted(matched_kw)) if matched_kw else "None")

            with col2:
                st.markdown("#### Missing Important Keywords")
                st.write(", ".join(sorted(missing_kw)) if missing_kw else "None")

            st.markdown("#### Section Check")
            for sec, present in section_presence.items():
                icon = "✅" if present else "❌"
                st.write(f"{icon} {sec}")
        else:
            st.info("Paste a Job Description above to get the ATS match score.")

        tips = generate_resume_tips(
            res_score, ats_score, section_presence, no_of_pages,
            level, missing_kw, extracted_skills, recommended_skills,
            is_scanned,
        )

        st.markdown("---")
        st.markdown("## Resume Tips 💡")
        for t in tips:
            st.markdown(f"- {t}")

        st.markdown("---")
        if resume_videos:
            st.markdown("### Bonus Video for Resume Writing Tips 📹")
            st.video(random.choice(resume_videos))
        if interview_videos:
            st.markdown("### Bonus Video for Interview Tips 💡")
            st.video(random.choice(interview_videos))

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            insert_data(
                name, email, res_score, ts, no_of_pages, field, level,
                str(extracted_skills), str(recommended_skills), str(rec_course),
            )
        except Exception as e:
            st.warning(f"Could not save data to database: {e}")

        st.markdown("### 📄 Extracted Text From Resume")
        st.text_area("", text, height=300)

    # ---------- ADMIN ----------
    else:
        st.markdown("## Smart Resume Analyzer – Admin Panel")
        st.markdown("### Welcome to Admin Side")

        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type="password")

        if st.button("Login"):
            if ad_user == "C_vision" and ad_password == "Baas@007":
                st.success("Welcome C_vision Data")

                if not os.path.exists("user_data.csv"):
                    st.warning("No user data found.")
                    return
                df = pd.read_csv("user_data.csv")
                if df.empty:
                   st.warning("No user data found.")
                   return

                st.markdown("### User's 👨‍💻 Data")
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Report",
                    data=csv_data,
                    file_name="user_data_report.csv",
                    mime="text/csv",
                )

                st.markdown("### 📈 Pie-Chart for Predicted Field Recommendations")
                field_counts = df["Field"].replace("", "Unknown").value_counts().reset_index()
                field_counts.columns = ["Field", "Count"]
                fig_field = px.pie(
                    field_counts, names="Field", values="Count",
                    title="Predicted Field according to the Skills",
                )
                st.plotly_chart(fig_field, use_container_width=True)

                st.markdown("### 📊 Pie-Chart for User's Experience Level")
                level_counts = df["Level"].value_counts().reset_index()
                level_counts.columns = ["Level", "Count"]
                fig_level = px.pie(
                    level_counts, names="Level", values="Count",
                    title="User's Experience Level",
                )
                st.plotly_chart(fig_level, use_container_width=True)
            else:
                st.error("Wrong ID or Password")


run()
